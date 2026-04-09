# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   RAG + SQL Agent Application                                               ║
║   LangChain 0.3+  ·  Groq (LLaMA / Gemma)  ·  SQLite  ·  Streamlit        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TWO MODES                                                                   ║
║  ──────────                                                                  ║
║  1. RAG Mode  — Upload PDFs / CSVs / TXT → index into ChromaDB →            ║
║               ask questions answered from your documents                    ║
║                                                                              ║
║  2. SQL Agent — Auto-creates a sample SQLite database (employees,           ║
║               products, orders) → ask questions in plain English            ║
║               → agent writes + runs SQL → shows results                    ║
║                                                                              ║
║  LANGCHAIN COMPONENTS USED                                                   ║
║  ─────────────────────────                                                   ║
║  • ChatGroq                  — LLM (open-source via Groq API)               ║
║  • HuggingFaceEmbeddings     — free local embeddings (no API key needed)   ║
║  • Chroma                    — in-memory vector store                       ║
║  • RecursiveCharacterTextSplitter — document chunking                      ║
║  • create_retrieval_chain    — RAG chain                                    ║
║  • create_stuff_documents_chain — document QA chain                        ║
║  • ChatPromptTemplate        — structured prompts                           ║
║  • RunnableWithMessageHistory — multi-turn memory                          ║
║  • ChatMessageHistory        — history store                               ║
║  • SQLDatabase               — LangChain SQLite wrapper                    ║
║  • create_sql_agent          — natural language → SQL agent                ║
║  • SQLDatabaseToolkit        — SQL tools for the agent                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Setup
─────
  1. pip install -r requirements.txt
  2. Create .env:   GROQ_API_KEY=gsk_your-key-here
  3. python -m streamlit run rag_sql_app.py
"""

from __future__ import annotations
import os
import sqlite3
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# RAG components
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# SQL Agent components
from langchain_community.utilities import SQLDatabase

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════

MODELS: dict[str, str] = {
    "LLaMA 3.1 8B  (fast)":     "llama-3.1-8b-instant",
    "LLaMA 3.3 70B (powerful)": "llama-3.3-70b-versatile",
    "LLaMA 3.1 70B (balanced)": "llama-3.1-70b-versatile",
    "Gemma 2 9B":               "gemma2-9b-it",
}

# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS  (free, runs locally — no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """
    HuggingFaceEmbeddings runs locally — no API key required.
    all-MiniLM-L6-v2 is fast, small (~90MB), and works great for RAG.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SQLITE DATABASE SETUP
#  Creates a sample database with 3 tables for the SQL Agent demo.
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = Path("sample_business.db")


def create_sample_database() -> None:
    """
    Create a sample SQLite database with realistic business data.
    Tables: employees, products, orders, order_items
    """
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS departments (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        budget      REAL NOT NULL,
        location    TEXT
    );

    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        department  TEXT NOT NULL,
        role        TEXT NOT NULL,
        salary      REAL NOT NULL,
        hire_date   TEXT NOT NULL,
        city        TEXT
    );

    CREATE TABLE IF NOT EXISTS products (
        id          INTEGER PRIMARY KEY,
        name        TEXT NOT NULL,
        category    TEXT NOT NULL,
        price       REAL NOT NULL,
        stock       INTEGER NOT NULL,
        supplier    TEXT
    );

    CREATE TABLE IF NOT EXISTS orders (
        id          INTEGER PRIMARY KEY,
        customer    TEXT NOT NULL,
        employee_id INTEGER,
        order_date  TEXT NOT NULL,
        total       REAL NOT NULL,
        status      TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees(id)
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id          INTEGER PRIMARY KEY,
        order_id    INTEGER,
        product_id  INTEGER,
        quantity    INTEGER NOT NULL,
        unit_price  REAL NOT NULL,
        FOREIGN KEY (order_id)   REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # Only insert if tables are empty
    if cur.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO departments VALUES (?,?,?,?)",
            [
                (1, "Engineering",  850000, "Bangalore"),
                (2, "Sales",        420000, "Mumbai"),
                (3, "Marketing",    310000, "Delhi"),
                (4, "HR",           180000, "Chennai"),
                (5, "Data Science", 620000, "Hyderabad"),
            ],
        )
        cur.executemany(
            "INSERT INTO employees VALUES (?,?,?,?,?,?,?)",
            [
                (1,  "Priya Sharma",    "Engineering",  "Senior Engineer",    95000, "2020-03-15", "Bangalore"),
                (2,  "Rahul Gupta",     "Engineering",  "Junior Engineer",    65000, "2022-07-01", "Bangalore"),
                (3,  "Ananya Singh",    "Data Science", "ML Engineer",       105000, "2019-11-20", "Hyderabad"),
                (4,  "Vikram Patel",    "Sales",        "Sales Manager",      78000, "2021-04-10", "Mumbai"),
                (5,  "Meera Nair",      "Marketing",    "Marketing Lead",     72000, "2020-09-05", "Delhi"),
                (6,  "Arjun Kumar",     "Engineering",  "Tech Lead",         115000, "2018-06-12", "Bangalore"),
                (7,  "Divya Menon",     "HR",           "HR Manager",         68000, "2021-02-28", "Chennai"),
                (8,  "Rohit Verma",     "Data Science", "Data Analyst",       82000, "2022-01-15", "Hyderabad"),
                (9,  "Sneha Joshi",     "Sales",        "Sales Executive",    55000, "2023-03-01", "Mumbai"),
                (10, "Kiran Reddy",     "Engineering",  "DevOps Engineer",    88000, "2020-08-22", "Bangalore"),
                (11, "Pooja Mishra",    "Marketing",    "Content Strategist", 60000, "2022-05-10", "Delhi"),
                (12, "Amit Bhatia",     "Data Science", "Senior Data Analyst",92000, "2019-07-14", "Hyderabad"),
            ],
        )
        cur.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?,?)",
            [
                (1,  "Laptop Pro 15",       "Electronics",  85000, 45, "TechCorp"),
                (2,  "Wireless Mouse",       "Accessories",   1200, 230,"TechCorp"),
                (3,  "Standing Desk",        "Furniture",    22000, 18, "OfficeMax"),
                (4,  "Noise-Cancel Headset", "Electronics",   8500, 67, "AudioPro"),
                (5,  "Office Chair Ergo",    "Furniture",    15000, 25, "OfficeMax"),
                (6,  "USB-C Hub 7-in-1",     "Accessories",   3500, 120,"TechCorp"),
                (7,  "Monitor 27\" 4K",      "Electronics",  42000, 32, "DisplayTech"),
                (8,  "Mechanical Keyboard",  "Accessories",   6500, 89, "TechCorp"),
                (9,  "Webcam HD 1080p",      "Electronics",   4200, 156,"CamPro"),
                (10, "Desk Lamp LED",        "Accessories",   1800, 200,"LightCo"),
            ],
        )
        cur.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?)",
            [
                (1,  "Acme Corp",         4, "2024-01-15", 93200, "Delivered"),
                (2,  "GlobalTech Ltd",    9, "2024-01-22", 47000, "Delivered"),
                (3,  "StartupHub",        4, "2024-02-05", 128500,"Delivered"),
                (4,  "MegaCorp India",    9, "2024-02-18", 22400, "Delivered"),
                (5,  "Innovate Inc",      4, "2024-03-01", 85000, "Delivered"),
                (6,  "TechWave",          9, "2024-03-15", 67800, "Processing"),
                (7,  "DataDriven Co",     4, "2024-04-02", 54000, "Shipped"),
                (8,  "CloudFirst Ltd",    9, "2024-04-20", 38500, "Processing"),
                (9,  "AI Ventures",       4, "2024-05-05", 112000,"Shipped"),
                (10, "Digital India Corp",9, "2024-05-22", 29700, "Pending"),
            ],
        )
        cur.executemany(
            "INSERT INTO order_items VALUES (?,?,?,?,?)",
            [
                (1,  1, 1, 1, 85000), (2,  1, 2, 2, 1200),  (3,  1, 4, 1, 8500),
                (4,  2, 7, 1, 42000), (5,  2, 6, 1, 3500),  (6,  3, 1, 1, 85000),
                (7,  3, 7, 1, 42000), (8,  4, 5, 1, 15000), (9,  4, 10,4, 1800),
                (10, 5, 1, 1, 85000), (11, 6, 3, 2, 22000), (12, 6, 8, 2, 6500),
                (13, 7, 9, 3, 4200),  (14, 7, 2, 5, 1200),  (15, 8, 10,5, 1800),
                (16, 9, 1, 1, 85000), (17, 9, 7, 1, 42000), (18,10, 2, 3, 1200),
            ],
        )

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str, file_type: str) -> list:
    """Load a document and return LangChain Document objects."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_type in ("txt", "md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


# ══════════════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
#
#  Flow:
#  Documents → Chunking → Embeddings → ChromaDB (vector store)
#                                           ↓
#  User Query → Embed Query → Similarity Search → Top-K Chunks
#                                           ↓
#  Chunks + Query → ChatPromptTemplate → ChatGroq → Answer
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorstore(documents: list) -> Chroma:
    """
    Split documents into chunks and index them into ChromaDB.

    RecursiveCharacterTextSplitter splits on paragraphs, then sentences,
    then words — trying to keep semantically related text together.
    chunk_overlap ensures context is not lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # characters per chunk
        chunk_overlap=150,     # overlap between consecutive chunks
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    # Chroma stores vectors in memory (persist_directory=None)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_documents",
    )
    return vectorstore, len(chunks)


def build_rag_chain(vectorstore: Chroma, model_id: str, temperature: float):
    """
    Build the RAG pipeline using pure LCEL (LangChain Expression Language).

    Flow:
      User query → retrieve top-4 chunks → format into prompt → LLM → answer

    No deprecated langchain.chains imports needed — uses only:
      langchain_core.prompts, langchain_core.messages, langchain_groq
    """
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model_id,
        temperature=temperature,
        max_tokens=2048,
    )

    # Retriever: find top 4 most similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Format retrieved docs into a single context string
    def format_docs(docs):
        return "\n\n".join(
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # QA prompt: system context + history + user question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions based ONLY on the "
         "provided document context below. If the answer is not in the context, "
         "say: \'I could not find that information in the uploaded documents.\'\n\n"
         "Document context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Pure LCEL chain — no deprecated imports
    # Returns both the answer and the source documents for citation display
    def run_rag(inputs: dict) -> dict:
        query        = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        docs         = retriever.invoke(query)
        context      = format_docs(docs)
        prompt_val   = qa_prompt.invoke({
            "context":      context,
            "chat_history": chat_history,
            "input":        query,
        })
        response = llm.invoke(prompt_val)
        return {
            "answer":  response.content,
            "context": docs,
            "input":   query,
        }

    return RunnableLambda(run_rag)


# ══════════════════════════════════════════════════════════════════════════════
#  SQL AGENT
#
#  Flow:
#  User Question (English)
#       ↓
#  SQLDatabaseToolkit  →  provides tools: list_tables, schema, query, checker
#       ↓
#  create_sql_agent  →  ReAct loop:
#       Thought → which table? → Action: list/inspect → Observation
#       Thought → write SQL  → Action: execute query → Observation
#       Thought → Final Answer (formatted result)
# ══════════════════════════════════════════════════════════════════════════════

def get_db_schema() -> str:
    """Read the SQLite schema and first few rows of each table."""
    conn = sqlite3.connect(str(DB_PATH))
    cur  = conn.cursor()
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    schema_parts = []
    for t in tables:
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        col_def = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        rows    = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
        rows_str = "\n".join(str(r) for r in rows)
        schema_parts.append(
            f"Table: {t}\nColumns: {col_def}\nSample rows:\n{rows_str}"
        )
    conn.close()
    return "\n\n".join(schema_parts)


def run_sql_query(sql: str) -> str:
    """Execute a SQL query and return results as a formatted string."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur  = conn.cursor()
        cur.execute(sql)
        rows    = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        if not rows:
            return "Query returned no results."
        # Format as a table
        col_widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
                      for i, c in enumerate(columns)]
        header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
        sep    = "-+-".join("-" * w for w in col_widths)
        body   = "\n".join(
            " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
            for row in rows
        )
        return f"{header}\n{sep}\n{body}\n\n({len(rows)} row(s) returned)"
    except Exception as e:
        return f"SQL Error: {e}"


SQL_SYSTEM_PROMPT = """You are an expert SQL assistant for a SQLite database.

Given a user question and the database schema below, you must:
1. Write a single correct SQLite SQL query to answer the question
2. Return ONLY the SQL query — no explanation, no markdown, no backticks
3. Use table and column names exactly as shown in the schema
4. For aggregations, always use aliases (e.g. SUM(total) AS total_revenue)
5. Always include ORDER BY and LIMIT where appropriate

DATABASE SCHEMA:
{schema}

Return only the SQL query, nothing else."""


def run_sql_chain(user_question: str, model_id: str, temperature: float) -> str:
    """
    Pure LCEL SQL chain — no AgentExecutor, no deprecated imports.

    Flow:
      1. Load schema from SQLite
      2. Ask LLM to generate SQL query
      3. Execute query against SQLite
      4. Ask LLM to format the results into a readable answer
    """
    create_sample_database()
    schema = get_db_schema()

    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model_id,
        temperature=0.0,   # Always 0 for SQL generation — needs to be deterministic
        max_tokens=512,
    )

    # Step 1: Generate SQL
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_SYSTEM_PROMPT),
        ("human", "Question: {question}"),
    ])
    sql_chain  = sql_prompt | llm
    sql_resp   = sql_chain.invoke({"schema": schema, "question": user_question})
    raw_sql    = sql_resp.content.strip()

    # Clean up any accidental markdown fences
    import re
    raw_sql = re.sub(r"```sql|```", "", raw_sql).strip()

    # Step 2: Execute SQL
    query_result = run_sql_query(raw_sql)

    # Step 3: Format result into natural language
    answer_llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model_id,
        temperature=temperature,
        max_tokens=1024,
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful data analyst. Given a SQL query result, "
         "provide a clear, friendly, well-formatted answer to the user's question. "
         "Use bullet points or numbered lists when listing multiple items. "
         "Include relevant numbers and format currency with commas."),
        ("human",
         "User question: {question}\n\n"
         "SQL used: {sql}\n\n"
         "Query result:\n{result}\n\n"
         "Please provide a clear answer:"),
    ])
    answer_chain  = answer_prompt | answer_llm
    final_answer  = answer_chain.invoke({
        "question": user_question,
        "sql":      raw_sql,
        "result":   query_result,
    })

    # Show the SQL used (helpful for learning)
    return f"{final_answer.content}\n\n---\n**SQL used:** `{raw_sql}`"


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_chat_history(key: str) -> ChatMessageHistory:
    if key not in st.session_state:
        st.session_state[key] = ChatMessageHistory()
    return st.session_state[key]


def get_display_history(key: str) -> list[dict]:
    dkey = f"disp_{key}"
    if dkey not in st.session_state:
        st.session_state[dkey] = []
    return st.session_state[dkey]


def clear_session(key: str) -> None:
    for k in [key, f"disp_{key}", f"vs_{key}"]:
        if k in st.session_state:
            del st.session_state[k]


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="RAG + SQL Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            padding: 2rem; border-radius: 14px; margin-bottom: 1.5rem; text-align: center;
        }
        .main-header h1 { color: #c084fc; font-size: 2.1rem; margin: 0; }
        .main-header p  { color: #a5b4fc; margin: 0.5rem 0 0 0; }
        .mode-badge {
            display: inline-block; padding: 4px 14px; border-radius: 20px;
            font-size: 0.82rem; font-weight: 700; margin-bottom: 0.8rem;
        }
        .rag-badge  { background: #4c1d95; color: #c084fc; border: 1px solid #7c3aed44; }
        .sql-badge  { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb44; }
        .info-card  {
            background: var(--secondary-background-color);
            border-left: 3px solid #7c3aed; border-radius: 6px;
            padding: 0.7rem 1rem; margin: 0.4rem 0; font-size: 0.88rem;
        }
        .sql-card {
            border-left: 3px solid #2563eb;
        }
        .schema-box {
            background: var(--secondary-background-color);
            border-radius: 8px; padding: 0.8rem; font-size: 0.8rem;
            font-family: monospace; margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🧠 RAG + SQL Agent</h1>
        <p>Retrieval Augmented Generation  ·  SQLite Agent  ·  LangChain 0.3+  ·  Groq Open-Source LLMs</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Mode selector
        mode = st.radio(
            "Select Mode:",
            ["📄 RAG — Document Q&A", "🗄️ SQL Agent — Database Q&A"],
            index=0,
        )
        is_rag = mode.startswith("📄")

        st.divider()

        # Model
        st.markdown("### 🧠 Model")
        model_label = st.selectbox("LLM:", list(MODELS.keys()))
        model_id    = MODELS[model_label]

        # Temperature
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                                help="Lower = more factual (recommended for RAG/SQL)")

        st.divider()

        # Session management
        session_key = f"{'rag' if is_rag else 'sql'}_{model_id}"
        history     = get_display_history(session_key)
        col1, col2  = st.columns(2)
        with col1:
            st.markdown(f"💬 **{len(history)}** messages")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                clear_session(session_key)
                st.rerun()

        st.divider()

        if is_rag:
            # ── RAG file uploader ─────────────────────────────────────────────
            st.markdown("### 📁 Upload Documents")
            uploaded = st.file_uploader(
                "PDF, CSV, or TXT files",
                type=["pdf", "csv", "txt", "md"],
                accept_multiple_files=True,
                help="Upload documents to ask questions about",
            )

            if uploaded:
                vs_key = f"vs_{session_key}"
                if vs_key not in st.session_state:
                    all_docs = []
                    with st.spinner("Reading and indexing documents..."):
                        for f in uploaded:
                            suffix = f.name.split(".")[-1].lower()
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=f".{suffix}"
                            ) as tmp:
                                tmp.write(f.read())
                                tmp_path = tmp.name
                            try:
                                docs = load_document(tmp_path, suffix)
                                all_docs.extend(docs)
                            except Exception as e:
                                st.error(f"Error loading {f.name}: {e}")
                            finally:
                                os.unlink(tmp_path)

                        if all_docs:
                            vs, n_chunks = build_vectorstore(all_docs)
                            st.session_state[vs_key] = vs
                            st.success(
                                f"✅ Indexed {len(all_docs)} pages → "
                                f"{n_chunks} chunks"
                            )

            with st.expander("🏗️ RAG Architecture", expanded=False):
                st.markdown("""
                **LangChain Components:**
                1. `PyPDFLoader / CSVLoader` — load docs
                2. `RecursiveCharacterTextSplitter` — chunk text
                3. `HuggingFaceEmbeddings` — embed chunks
                4. `Chroma` — vector store (similarity search)
                5. `create_retrieval_chain` — RAG pipeline
                6. `create_stuff_documents_chain` — doc QA
                7. `ChatGroq` — answer generation
                """)
        else:
            # ── SQL info ──────────────────────────────────────────────────────
            st.markdown("### 🗄️ Database")
            create_sample_database()
            conn = sqlite3.connect(str(DB_PATH))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn.close()
            st.markdown("**Tables:**")
            for t in tables:
                st.markdown(f"  - `{t}`")

            with st.expander("🏗️ SQL Agent Architecture", expanded=False):
                st.markdown("""
                **LangChain Components:**
                1. `SQLDatabase` — SQLite wrapper
                2. `SQLDatabaseToolkit` — list/inspect/query tools
                3. `create_sql_agent` — ReAct SQL agent
                4. `ChatGroq` — LLM reasoning engine

                **Agent loop:**
                Question → Inspect schema → Write SQL
                → Execute → Format answer
                """)

        if not os.environ.get("GROQ_API_KEY"):
            st.error("⚠️ GROQ_API_KEY missing!\n.env: GROQ_API_KEY=gsk_...")

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    if is_rag:
        st.markdown('<div class="mode-badge rag-badge">📄 RAG — Document Q&A Mode</div>',
                    unsafe_allow_html=True)

        vs_key = f"vs_{session_key}"
        if vs_key not in st.session_state:
            st.info(
                "👈 **Upload documents** in the sidebar to get started.\n\n"
                "Supported: **PDF**, **CSV**, **TXT**, **Markdown**"
            )
            # Show example questions
            st.markdown("#### Example questions after uploading:")
            for q in [
                "What is the main topic of this document?",
                "Summarise the key findings.",
                "What does the document say about [topic]?",
            ]:
                st.markdown(f"- *{q}*")
            return

        # Show chat history
        history = get_display_history(session_key)
        if not history:
            with st.chat_message("assistant", avatar="🧠"):
                st.markdown(
                    "Documents indexed! Ask me anything about your uploaded files."
                )

        for msg in history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Source chunks used", expanded=False):
                        st.markdown(msg["sources"])

        user_q = st.chat_input("Ask a question about your documents...")
        if user_q:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_q)
            history.append({"role": "user", "content": user_q})

            with st.chat_message("assistant", avatar="🧠"):
                ph = st.empty()
                try:
                    with st.spinner("Searching documents..."):
                        vs  = st.session_state[vs_key]
                        chain = build_rag_chain(vs, model_id, temperature)
                        chat_hist = get_chat_history(session_key)

                        result = chain.invoke({
                            "input":        user_q,
                            "chat_history": chat_hist.messages,
                        })
                        answer = result.get("answer", "No answer generated.")

                        # Format source references
                        source_docs = result.get("context", [])
                        sources_md  = ""
                        for i, doc in enumerate(source_docs, 1):
                            meta   = doc.metadata
                            source = meta.get("source", "uploaded file")
                            page   = meta.get("page", "")
                            page_s = f" (page {page+1})" if page != "" else ""
                            snip   = doc.page_content[:200].replace("\n", " ")
                            sources_md += f"**Chunk {i}** — {Path(source).name}{page_s}\n> {snip}...\n\n"

                        # Update memory
                        from langchain_core.messages import HumanMessage, AIMessage
                        chat_hist.add_message(HumanMessage(content=user_q))
                        chat_hist.add_message(AIMessage(content=answer))

                    ph.markdown(answer)

                except Exception as e:
                    answer = f"❌ Error: {e}"
                    ph.error(answer)
                    sources_md = ""

            history.append({
                "role":    "assistant",
                "content": answer,
                "sources": sources_md,
            })
            st.session_state[f"disp_{session_key}"] = history
            st.rerun()

    else:
        # ── SQL AGENT MODE ────────────────────────────────────────────────────
        st.markdown('<div class="mode-badge sql-badge">🗄️ SQL Agent — Database Q&A Mode</div>',
                    unsafe_allow_html=True)

        # Database schema preview
        with st.expander("📊 View Database Schema", expanded=False):
            create_sample_database()
            conn   = sqlite3.connect(str(DB_PATH))
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            for tname in tables:
                cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
                col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
                count   = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                st.markdown(
                    f'<div class="schema-box"><b>{tname}</b> ({count} rows)<br>'
                    f'<span style="color:gray">{col_str}</span></div>',
                    unsafe_allow_html=True,
                )
            conn.close()

        # Example questions
        st.markdown("#### 💡 Try these questions:")
        ex_cols = st.columns(2)
        examples = [
            "Who are the top 3 highest paid employees?",
            "What is the total revenue by month in 2024?",
            "Which products are low on stock (less than 30 units)?",
            "What is the average salary per department?",
            "List all pending or processing orders with their totals.",
            "Which employee has handled the most orders?",
        ]
        for i, ex in enumerate(examples):
            col = ex_cols[i % 2]
            with col:
                if st.button(f"💬 {ex}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["sql_prefill"] = ex

        st.markdown("---")

        # Chat display
        history = get_display_history(session_key)
        if not history:
            with st.chat_message("assistant", avatar="🗄️"):
                st.markdown(
                    "Database ready! Ask me anything about the business data. "
                    "I'll write and run SQL automatically."
                )

        for msg in history:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🗄️"):
                st.markdown(msg["content"])

        prefill    = st.session_state.pop("sql_prefill", "")
        user_input = st.chat_input("Ask a question about the database...") or prefill

        if user_input:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)
            history.append({"role": "user", "content": user_input})

            with st.chat_message("assistant", avatar="🗄️"):
                ph = st.empty()
                try:
                    with st.spinner("Writing and running SQL..."):
                        answer = run_sql_chain(user_input, model_id, temperature)
                    ph.markdown(answer)
                except Exception as e:
                    answer = f"❌ SQL Agent error: {e}"
                    ph.error(answer)

            history.append({"role": "assistant", "content": answer})
            st.session_state[f"disp_{session_key}"] = history
            st.rerun()


if __name__ == "__main__":
    main()
