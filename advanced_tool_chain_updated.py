"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ADVANCED TOOL-CHAINING MULTI-AGENT SYSTEM — LangGraph + Groq             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ARCHITECTURE (Mixed Sequential + Parallel with Tool Chaining):             ║
║                                                                              ║
║   ┌─────────────────────────────────────────────────────┐                   ║
║   │  🧭 AGENT 1 — Query Router                         │  SEQUENTIAL        ║
║   │  Tools: LLM Parser                                  │                   ║
║   │  Output: topic, company, ticker, city → shared state│                   ║
║   └──────────────────────────┬──────────────────────────┘                   ║
║                              │ fan-out (3 parallel branches)                 ║
║          ┌───────────────────┼────────────────────┐                         ║
║          ▼                   ▼                    ▼                         ║
║  ┌───────────────┐  ┌────────────────┐  ┌─────────────────┐                 ║
║  │ 📰 AGENT 2   │  │ 📈 AGENT 3    │  │ 🌦️ AGENT 4      │  PARALLEL        ║
║  │ News Fetcher  │  │ Finance Agent  │  │ Weather Agent    │                 ║
║  │ Tools:        │  │ Tools:         │  │ Tools:           │                 ║
║  │ ① Google RSS  │  │ ① yFinance API │  │ ① Geocode API    │                 ║
║  │    ↓ chain    │  │    ↓ chain     │  │    ↓ chain       │                 ║
║  │ ② DuckDuckGo  │  │ ② SQLite DB    │  │ ② Weather API    │                 ║
║  └──────┬────────┘  └──────┬─────────┘  └──────┬──────────┘                 ║
║         │                  │                    │  fan-in                    ║
║         └──────────────────┼────────────────────┘                           ║
║                            ▼                                                 ║
║   ┌─────────────────────────────────────────────────────┐                   ║
║   │  🗄️ AGENT 5 — SQL Intelligence Agent               │  SEQUENTIAL        ║
║   │  Tools: SQLite queries (reads DB written by Agent 3) │                   ║
║   │  Chain: DB written by Agent 3 → read by Agent 5     │                   ║
║   └──────────────────────────┬──────────────────────────┘                   ║
║                              ▼                                               ║
║   ┌─────────────────────────────────────────────────────┐                   ║
║   │  🧠 AGENT 6 — Master Synthesiser                   │  SEQUENTIAL        ║
║   │  Tools: Wikipedia API + LLM                         │                   ║
║   │  Chain: ALL prior tool outputs → final report       │                   ║
║   └─────────────────────────────────────────────────────┘                   ║
║                                                                              ║
║  TOOL CHAIN MAP:                                                             ║
║   Google RSS  ──→ DuckDuckGo   (news chain: RSS headlines → deep search)    ║
║   yFinance    ──→ SQLite Write  (finance chain: API fetch → DB store)       ║
║   SQLite Write──→ SQLite Read   (DB chain: Agent 3 writes → Agent 5 reads)  ║
║   Geocode API ──→ Weather API   (geo chain: city→coords → forecast)         ║
║   All outputs ──→ Wikipedia     (synthesis chain: context → enrichment)     ║
║   Wikipedia   ──→ LLM Report    (final chain: all sources → report)         ║
║                                                                              ║
║  LLM  : llama-3.3-70b-versatile via Groq (FREE tier)                        ║
║  DB   : SQLite (in-memory, persisted to file)                               ║
║  UI   : Streamlit                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import sqlite3
import operator
import requests
import xml.etree.ElementTree as ET
from typing   import TypedDict, Annotated, Optional
from datetime import datetime
from dotenv   import load_dotenv

from langchain_groq       import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph      import StateGraph, END
from duckduckgo_search    import DDGS

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# DB PATH  (SQLite file created in same folder as script)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "intelligence.db")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
#  Annotated[list, operator.add]  →  safe for parallel fan-in (append-only)
# ══════════════════════════════════════════════════════════════════════════════

class IntelState(TypedDict):
    # ── User input ───────────────────────────────────────────────
    user_query:       str

    # ── Agent 1 parsed entities ──────────────────────────────────
    company:          Optional[str]
    ticker:           Optional[str]
    topic:            Optional[str]
    city:             Optional[str]
    country:          Optional[str]

    # ── Parallel agent outputs (fan-in accumulators) ─────────────
    parallel_outputs: Annotated[list[str], operator.add]

    # ── Sequential agent outputs ─────────────────────────────────
    sql_intelligence: Optional[str]
    final_report:     Optional[str]

    # ── Execution log (accumulator) ──────────────────────────────
    logs:             Annotated[list[str], operator.add]


# ══════════════════════════════════════════════════════════════════════════════
#  LLM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def llm(temp: float = 0.3, tokens: int = 1500) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temp,
        max_tokens=tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ┌─────────────────────────────────────────────────────────────┐
#  │              TOOL LIBRARY  (6 custom tools)                 │
#  └─────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════

# ─── TOOL 1 ─── LLM Query Parser ────────────────────────────────────────────

def tool_parse_query(raw: str) -> dict:
    """
    TOOL: LLM Query Parser
    ────────────────────────
    Calls Groq LLM to extract structured entities from a free-text query.
    Returns: company, ticker, topic, city, country
    """
    response = llm(temp=0).invoke([
        SystemMessage(content=(
            "Extract entities from the user query. "
            "Reply in valid JSON only — no markdown fences, no preamble.\n"
            "Schema: {\"company\":\"...\",\"ticker\":\"...\",\"topic\":\"...\","
            "\"city\":\"...\",\"country\":\"...\"}\n"
            "Rules:\n"
            "- ticker: known stock symbol or UNKNOWN\n"
            "- city: HQ city or the city explicitly mentioned\n"
            "- If unsure, make the most reasonable inference.\n"
            "- All values must be non-empty strings."
        )),
        HumanMessage(content=f"User query: {raw}"),
    ])
    try:
        text = response.content.strip()
        # strip accidental markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"company": raw, "ticker": "UNKNOWN",
                "topic": raw, "city": "Mumbai", "country": "India"}


# ─── TOOL 2 ─── Google News RSS ──────────────────────────────────────────────

def tool_google_news_rss(query: str, max_items: int = 8) -> list[dict]:
    """
    TOOL: Google News RSS
    ──────────────────────
    Calls the public Google News RSS feed — no API key required.
    Returns a list of {title, source, published, link} dicts.

    Chain: output is passed directly into tool_duckduckgo_search as context.
    """
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title  = (item.findtext("title")       or "").strip()
            source = (item.findtext("source")      or "Google News").strip()
            pub    = (item.findtext("pubDate")      or "").strip()
            link   = (item.findtext("link")         or "").strip()
            items.append({"title": title, "source": source,
                          "published": pub, "link": link})
        return items if items else [{"title": f"No RSS results for '{query}'",
                                     "source": "", "published": "", "link": ""}]
    except Exception as e:
        return [{"title": f"RSS error: {str(e)}", "source": "",
                 "published": "", "link": ""}]


# ─── TOOL 3 ─── DuckDuckGo Deep Search ───────────────────────────────────────

def tool_duckduckgo_search(query: str, max_results: int = 6) -> str:
    """
    TOOL: DuckDuckGo Web Search
    ────────────────────────────
    Performs a deep-web search. No API key required.
    Receives RSS headline context from Tool 2 as part of the query.

    Chain: called AFTER tool_google_news_rss to do deeper research.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No DuckDuckGo results for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}\n    {r['body'][:200]}\n    {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"DuckDuckGo error: {str(e)}"


# ─── TOOL 4 ─── yFinance / Open Finance API ──────────────────────────────────

def tool_fetch_financials(ticker: str, company: str) -> dict:
    """
    TOOL: Yahoo Finance API (via yfinance library)
    ────────────────────────────────────────────────
    Fetches live stock metrics for the given ticker.
    Falls back to a mock dataset if yfinance not installed.

    Chain: output dict is stored in SQLite by tool_sql_write.
    Returns: price, market_cap, pe_ratio, week52_high/low, volume, etc.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="5d")
        price  = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None and not hist.empty:
            price = round(float(hist["Close"].iloc[-1]), 2)
        return {
            "company":       company,
            "ticker":        ticker,
            "price":         price,
            "currency":      info.get("currency", "USD"),
            "market_cap":    info.get("marketCap"),
            "pe_ratio":      info.get("trailingPE"),
            "week52_high":   info.get("fiftyTwoWeekHigh"),
            "week52_low":    info.get("fiftyTwoWeekLow"),
            "volume":        info.get("volume"),
            "avg_volume":    info.get("averageVolume"),
            "sector":        info.get("sector",   "N/A"),
            "industry":      info.get("industry", "N/A"),
            "employees":     info.get("fullTimeEmployees"),
            "description":   (info.get("longBusinessSummary") or "")[:400],
            "fetch_ts":      datetime.utcnow().isoformat(),
            "source":        "yfinance / Yahoo Finance API",
        }
    except ImportError:
        # ── Mock fallback if yfinance not installed ──────────────
        import random, math
        base  = random.uniform(80, 3500)
        return {
            "company":    company, "ticker": ticker,
            "price":      round(base, 2),
            "currency":   "USD",
            "market_cap": int(base * 1_000_000 * random.uniform(500, 8000)),
            "pe_ratio":   round(random.uniform(12, 45), 2),
            "week52_high":round(base * random.uniform(1.1, 1.6), 2),
            "week52_low": round(base * random.uniform(0.5, 0.9), 2),
            "volume":     int(random.uniform(2_000_000, 80_000_000)),
            "avg_volume": int(random.uniform(5_000_000, 60_000_000)),
            "sector":     "Technology", "industry": "Software",
            "employees":  int(random.uniform(5000, 200000)),
            "description": f"Mock financial data for {company} ({ticker}). "
                           "Install yfinance for real data: pip install yfinance",
            "fetch_ts":   datetime.utcnow().isoformat(),
            "source":     "Mock data (install yfinance for live data)",
        }
    except Exception as e:
        return {"company": company, "ticker": ticker, "error": str(e),
                "fetch_ts": datetime.utcnow().isoformat(), "source": "yfinance error"}


# ─── TOOL 5 ─── SQLite Database (Write + Read) ───────────────────────────────

def tool_sql_write(data: dict) -> str:
    """
    TOOL: SQLite Database Writer
    ──────────────────────────────
    Creates the 'financial_snapshots' table if needed, then
    INSERT the financial data fetched by Tool 4.

    Chain input:  output dict from tool_fetch_financials
    Chain output: confirmation string used by Agent 5 to query the DB.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts       TEXT,
            company      TEXT,
            ticker       TEXT,
            price        REAL,
            currency     TEXT,
            market_cap   REAL,
            pe_ratio     REAL,
            week52_high  REAL,
            week52_low   REAL,
            volume       INTEGER,
            avg_volume   INTEGER,
            sector       TEXT,
            industry     TEXT,
            employees    INTEGER,
            source       TEXT
        )
    """)
    cur.execute("""
        INSERT INTO financial_snapshots
        (run_ts, company, ticker, price, currency, market_cap, pe_ratio,
         week52_high, week52_low, volume, avg_volume, sector, industry, employees, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("fetch_ts"),
        data.get("company"),    data.get("ticker"),
        data.get("price"),      data.get("currency"),
        data.get("market_cap"), data.get("pe_ratio"),
        data.get("week52_high"),data.get("week52_low"),
        data.get("volume"),     data.get("avg_volume"),
        data.get("sector"),     data.get("industry"),
        data.get("employees"),  data.get("source"),
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return f"✅ SQL WRITE: Row #{row_id} inserted into financial_snapshots at {DB_PATH}"


def tool_sql_query(sql: str) -> list[dict]:
    """
    TOOL: SQLite Database Reader / Query Engine
    ────────────────────────────────────────────
    Executes any SELECT query against the local SQLite DB.

    Chain input:  DB populated by tool_sql_write (Agent 3)
    Chain output: list of row dicts consumed by Agent 5 for intelligence.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return [{"error": str(e), "sql": sql}]


# ─── TOOL 6 ─── Geocoding → Weather Chain ────────────────────────────────────

def tool_geocode(city: str) -> dict:
    """
    TOOL: Open-Meteo Geocoding API
    ────────────────────────────────
    Converts a city name to lat/lon + metadata.
    Chain output: coordinates are fed directly into tool_weather_forecast.
    """
    try:
        data = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=8
        ).json()
        r = data.get("results", [{}])[0]
        return {
            "name":      r.get("name", city),
            "country":   r.get("country", ""),
            "lat":       r.get("latitude"),
            "lon":       r.get("longitude"),
            "timezone":  r.get("timezone", ""),
            "elevation": r.get("elevation"),
        }
    except Exception as e:
        return {"name": city, "country": "", "lat": 19.07, "lon": 72.87,
                "timezone": "Asia/Kolkata", "elevation": 14, "error": str(e)}


def tool_weather_forecast(lat: float, lon: float, city_name: str) -> dict:
    """
    TOOL: Open-Meteo Weather Forecast API
    ────────────────────────────────────────
    Fetches current weather + 3-day forecast using coordinates from geocoding.
    Chain input: lat/lon from tool_geocode
    """
    WMO = {
        0:"☀️ Clear sky",1:"🌤️ Mainly clear",2:"⛅ Partly cloudy",3:"☁️ Overcast",
        45:"🌫️ Fog",51:"🌦️ Light drizzle",61:"🌧️ Light rain",63:"🌧️ Rain",
        65:"🌧️ Heavy rain",71:"❄️ Light snow",80:"🌦️ Showers",95:"⛈️ Thunderstorm",
    }
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current_weather": "true",
                "hourly": "relative_humidity_2m,apparent_temperature,precipitation_probability",
                "daily":  "temperature_2m_max,temperature_2m_min,weathercode",
                "forecast_days": 4,
                "timezone": "auto",
            },
            timeout=8
        ).json()
        cw  = resp.get("current_weather", {})
        hr  = resp.get("hourly", {})
        day = resp.get("daily",  {})
        forecast = []
        dates = day.get("time", [])
        for i in range(min(4, len(dates))):
            forecast.append({
                "date":    dates[i],
                "max_c":   day.get("temperature_2m_max", [None])[i],
                "min_c":   day.get("temperature_2m_min", [None])[i],
                "condition": WMO.get(day.get("weathercode", [0])[i], "Unknown"),
            })
        return {
            "city":       city_name,
            "temp_c":     cw.get("temperature"),
            "feels_like": hr.get("apparent_temperature", [None])[0],
            "condition":  WMO.get(cw.get("weathercode", 0), "Unknown"),
            "humidity":   hr.get("relative_humidity_2m", [None])[0],
            "rain_pct":   hr.get("precipitation_probability", [None])[0],
            "windspeed":  cw.get("windspeed"),
            "forecast":   forecast,
        }
    except Exception as e:
        return {"city": city_name, "error": str(e)}


# ─── TOOL 7 ─── Wikipedia ────────────────────────────────────────────────────

def tool_wikipedia(topic: str) -> str:
    """
    TOOL: Wikipedia REST API
    ──────────────────────────
    Fetches the summary extract for a topic. Falls back to search.
    Chain: called by Agent 6 to add background context to the final report.
    """
    try:
        slug = topic.strip().replace(" ", "_")
        url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        r    = requests.get(url, timeout=8,
                            headers={"User-Agent": "IntelAgent/1.0"})
        if r.status_code != 200:
            # fallback: opensearch
            s = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action":"opensearch","search":topic,"limit":1,"format":"json"},
                timeout=6
            ).json()
            if s[1]:
                slug = s[1][0].replace(" ", "_")
                r    = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
                    timeout=8, headers={"User-Agent": "IntelAgent/1.0"}
                )
        d = r.json()
        return (
            f"📖 Wikipedia — {d.get('title','')}\n\n"
            f"{d.get('extract','No summary found.')}\n\n"
            f"🔗 {d.get('content_urls',{}).get('desktop',{}).get('page','')}"
        )
    except Exception as e:
        return f"Wikipedia error for '{topic}': {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  ┌─────────────────────────────────────────────────────────────┐
#  │                    AGENT NODES                              │
#  └─────────────────────────────────────────────────────────────┘
# ══════════════════════════════════════════════════════════════════════════════

# ─── AGENT 1 ─── Query Router  (SEQUENTIAL) ──────────────────────────────────

def agent_query_router(state: IntelState) -> dict:
    """
    AGENT 1 — Query Router
    ════════════════════════
    Tool  : LLM Query Parser (Tool 1)
    Role  : Parses the raw user query into structured entities used by
            all downstream agents.
    Output: company, ticker, topic, city, country → written to shared state
    Mode  : SEQUENTIAL (entry point)

    ✅ May use **state — runs alone as the entry node.
    """
    print("\n🧭 [Agent 1 — Query Router] Parsing query...")
    parsed = tool_parse_query(state["user_query"])
    company = parsed.get("company", state["user_query"])
    ticker  = parsed.get("ticker",  "UNKNOWN")
    topic   = parsed.get("topic",   state["user_query"])
    city    = parsed.get("city",    "Mumbai")
    country = parsed.get("country", "India")
    print(f"   ✓ company={company} | ticker={ticker} | city={city}")
    return {
        **state,
        "company": company,
        "ticker":  ticker,
        "topic":   topic,
        "city":    city,
        "country": country,
        "logs": [f"🧭 Agent 1 [Query Router]: company='{company}' ticker='{ticker}' city='{city}'"],
    }


# ─── AGENT 2 ─── News Researcher  (PARALLEL) ─────────────────────────────────

def agent_news_researcher(state: IntelState) -> dict:
    """
    AGENT 2 — News Researcher
    ══════════════════════════
    Tools (CHAINED):
      ① Google News RSS  → fetch latest headlines
         ↓ pass top headline as context
      ② DuckDuckGo       → deep-dive search enriched with RSS context
    Role  : Produces a combined news intelligence brief.
    Mode  : PARALLEL (fan-out branch)

    ⚠️ Returns ONLY its own keys — no **state spread.
    """
    print("\n📰 [Agent 2 — News Researcher] Running (PARALLEL)...")
    company = state.get("company", "")
    topic   = state.get("topic",   "")

    # ── CHAIN STEP 1: Google News RSS ────────────────────────────
    rss_items = tool_google_news_rss(f"{company} {topic} news 2025")
    rss_lines = []
    for i, item in enumerate(rss_items, 1):
        pub = item["published"][:16] if item["published"] else ""
        rss_lines.append(f"  {i:02d}. [{item['source']}] {item['title']} ({pub})")
    rss_block = "\n".join(rss_lines)
    top_headline = rss_items[0]["title"] if rss_items else company
    print(f"   ✓ Google RSS: {len(rss_items)} headlines fetched")

    # ── CHAIN STEP 2: DuckDuckGo — enriched with RSS context ─────
    ddg_query   = f"{company} {top_headline[:60]} latest analysis 2025"
    ddg_results = tool_duckduckgo_search(ddg_query, max_results=5)
    print(f"   ✓ DuckDuckGo: deep search complete")

    # ── LLM: synthesise both chained outputs ─────────────────────
    prompt = f"""You are a financial news intelligence analyst.
Below are TWO chained data sources about: {company} / {topic}

═══ SOURCE 1 — GOOGLE NEWS RSS HEADLINES ═══
{rss_block}

═══ SOURCE 2 — DUCKDUCKGO DEEP SEARCH ═══
{ddg_results}

Write a structured news intelligence brief with these exact sections:

📰 BREAKING HEADLINES (top 5 from RSS, with source and date)
🔍 DEEP WEB INTELLIGENCE (5 key findings from DuckDuckGo)
🔗 CHAINED SYNTHESIS (3–4 cross-source patterns/signals you identified)
⚠️ KEY RISKS & OPPORTUNITIES (bullet points)

Be specific. Cite sources. Max 500 words."""

    brief = llm(temp=0.3).invoke([
        SystemMessage(content="You are a sharp, precise financial news analyst."),
        HumanMessage(content=prompt),
    ]).content
    print(f"   ✓ News brief ready ({len(brief)} chars)")

    output = (
        "╔═══════════════════════════════════════╗\n"
        "║   📰  NEWS INTELLIGENCE  (Agent 2)   ║\n"
        "║   Tools: Google RSS → DuckDuckGo     ║\n"
        "╚═══════════════════════════════════════╝\n\n"
        f"RAW RSS HEADLINES:\n{rss_block}\n\n"
        f"CHAINED ANALYSIS:\n{brief}"
    )
    return {
        "parallel_outputs": [output],
        "logs": [f"📰 Agent 2 [News Researcher]: RSS({len(rss_items)} items) → DDG chained → brief generated"],
    }


# ─── AGENT 3 ─── Finance & SQL Agent  (PARALLEL) ─────────────────────────────

def agent_finance_sql(state: IntelState) -> dict:
    """
    AGENT 3 — Finance & SQL Agent
    ══════════════════════════════
    Tools (CHAINED):
      ① Yahoo Finance API  → fetch live financial metrics
         ↓ pass data dict
      ② SQLite Write Tool  → store fetched data in DB
    Role  : Fetches financial data and persists it to SQL for Agent 5.
    Mode  : PARALLEL (fan-out branch)

    ⚠️ Returns ONLY its own keys — no **state spread.
    """
    print("\n📈 [Agent 3 — Finance & SQL] Running (PARALLEL)...")
    company = state.get("company", "Unknown")
    ticker  = state.get("ticker",  "UNKNOWN")
    if ticker == "UNKNOWN":
        ticker = company[:4].upper()

    # ── CHAIN STEP 1: Yahoo Finance API ──────────────────────────
    fin_data = tool_fetch_financials(ticker, company)
    print(f"   ✓ Finance API: {ticker} data fetched (source: {fin_data.get('source','')})")

    # ── CHAIN STEP 2: Store in SQLite ────────────────────────────
    sql_confirm = tool_sql_write(fin_data)
    print(f"   ✓ SQL Write: {sql_confirm}")

    # ── Format the chained output ─────────────────────────────────
    def fmt(v):
        if v is None: return "N/A"
        if isinstance(v, float): return f"{v:,.2f}"
        if isinstance(v, int) and v > 1_000_000:
            return f"{v/1_000_000_000:.2f}B" if v > 1e9 else f"{v/1_000_000:.1f}M"
        return str(v)

    output = (
        "╔══════════════════════════════════════════╗\n"
        "║  📈  FINANCE + SQL CHAIN  (Agent 3)     ║\n"
        "║  Tools: yFinance API → SQLite Write     ║\n"
        "╚══════════════════════════════════════════╝\n\n"
        f"COMPANY      : {fin_data.get('company')} ({fin_data.get('ticker')})\n"
        f"DATA SOURCE  : {fin_data.get('source')}\n"
        f"FETCH TIME   : {fin_data.get('fetch_ts')}\n\n"
        "── LIVE MARKET DATA (from API) ──────────\n"
        f"  Price          : {fmt(fin_data.get('price'))} {fin_data.get('currency','')}\n"
        f"  Market Cap     : {fmt(fin_data.get('market_cap'))}\n"
        f"  P/E Ratio      : {fmt(fin_data.get('pe_ratio'))}\n"
        f"  52-Week High   : {fmt(fin_data.get('week52_high'))}\n"
        f"  52-Week Low    : {fmt(fin_data.get('week52_low'))}\n"
        f"  Volume         : {fmt(fin_data.get('volume'))}\n"
        f"  Avg Volume     : {fmt(fin_data.get('avg_volume'))}\n"
        f"  Sector         : {fin_data.get('sector')}\n"
        f"  Industry       : {fin_data.get('industry')}\n"
        f"  Employees      : {fmt(fin_data.get('employees'))}\n\n"
        "── SQL WRITE CONFIRMATION ────────────────\n"
        f"  {sql_confirm}\n"
        f"  DB Path: {DB_PATH}\n"
    )
    return {
        "parallel_outputs": [output],
        "logs": [f"📈 Agent 3 [Finance+SQL]: yFinance({ticker}) → SQLite write chained → row stored"],
    }


# ─── AGENT 4 ─── Weather & Geo Agent  (PARALLEL) ─────────────────────────────

def agent_weather_geo(state: IntelState) -> dict:
    """
    AGENT 4 — Weather & Geo Agent
    ══════════════════════════════
    Tools (CHAINED):
      ① Geocoding API  → city name → lat/lon + metadata
         ↓ pass coordinates
      ② Weather API    → lat/lon → current weather + 4-day forecast
    Role  : Produces weather + geo intelligence with business context.
    Mode  : PARALLEL (fan-out branch)

    ⚠️ Returns ONLY its own keys — no **state spread.
    """
    print("\n🌦️  [Agent 4 — Weather & Geo] Running (PARALLEL)...")
    city    = state.get("city",    "Mumbai")
    company = state.get("company", "")
    topic   = state.get("topic",   "")

    # ── CHAIN STEP 1: Geocoding ───────────────────────────────────
    geo = tool_geocode(city)
    print(f"   ✓ Geocode: {geo.get('name')}, {geo.get('country')} → ({geo.get('lat')}, {geo.get('lon')})")

    # ── CHAIN STEP 2: Weather using chained lat/lon ───────────────
    lat = geo.get("lat") or 19.07
    lon = geo.get("lon") or 72.87
    wx  = tool_weather_forecast(lat, lon, geo.get("name", city))
    print(f"   ✓ Weather: {wx.get('condition','?')} | {wx.get('temp_c','?')}°C")

    # ── LLM: business interpretation of weather context ──────────
    forecast_text = ""
    for f in wx.get("forecast", []):
        forecast_text += f"  {f['date']}: {f['condition']} | {f['min_c']}°C–{f['max_c']}°C\n"

    biz_prompt = (
        f"City: {geo.get('name')}, {geo.get('country')} | Company: {company} | Topic: {topic}\n"
        f"Current weather: {wx.get('condition')}, {wx.get('temp_c')}°C, "
        f"humidity {wx.get('humidity')}%, rain {wx.get('rain_pct')}%, wind {wx.get('windspeed')} km/h\n"
        f"4-day forecast:\n{forecast_text}\n\n"
        "In 4 concise bullet points, explain the BUSINESS IMPACT of these weather conditions "
        "specifically for this company/industry. Consider: supply chain, consumer behaviour, "
        "logistics, energy demand, operational risk, or seasonal demand."
    )
    biz_impact = llm(temp=0.4).invoke([
        SystemMessage(content="You connect weather conditions to business consequences. Be specific and insightful."),
        HumanMessage(content=biz_prompt),
    ]).content

    output = (
        "╔═══════════════════════════════════════════╗\n"
        "║  🌦️  WEATHER + GEO CHAIN  (Agent 4)      ║\n"
        "║  Tools: Geocoding API → Weather API      ║\n"
        "╚═══════════════════════════════════════════╝\n\n"
        "── GEO (from Geocoding API) ──────────────\n"
        f"  City       : {geo.get('name')}, {geo.get('country')}\n"
        f"  Coordinates: {geo.get('lat')}, {geo.get('lon')}\n"
        f"  Timezone   : {geo.get('timezone')}\n"
        f"  Elevation  : {geo.get('elevation')} m\n\n"
        "── CURRENT WEATHER (chained from coords) ─\n"
        f"  Condition  : {wx.get('condition')}\n"
        f"  Temperature: {wx.get('temp_c')}°C (feels {wx.get('feels_like')}°C)\n"
        f"  Humidity   : {wx.get('humidity')}%\n"
        f"  Rain chance: {wx.get('rain_pct')}%\n"
        f"  Wind       : {wx.get('windspeed')} km/h\n\n"
        "── 4-DAY FORECAST ────────────────────────\n"
        f"{forecast_text}\n"
        "── BUSINESS WEATHER IMPACT ───────────────\n"
        f"{biz_impact}"
    )
    return {
        "parallel_outputs": [output],
        "logs": [f"🌦️ Agent 4 [Weather+Geo]: Geocode({city}) → WeatherAPI chained → biz impact generated"],
    }


# ─── AGENT 5 ─── SQL Intelligence Agent  (SEQUENTIAL) ────────────────────────

def agent_sql_intelligence(state: IntelState) -> dict:
    """
    AGENT 5 — SQL Intelligence Agent
    ══════════════════════════════════
    Tool  : SQLite Query Engine (Tool: tool_sql_query)
    Role  : Reads the DB that Agent 3 wrote, runs 3 analytical queries,
            surfaces patterns and anomalies.
    Mode  : SEQUENTIAL (fan-in — waits for all parallel agents to finish)
    Chain : DB written by Agent 3 → read & analysed by Agent 5

    ✅ May use **state — runs sequentially after fan-in.
    """
    print("\n🗄️  [Agent 5 — SQL Intelligence] Running (sequential fan-in)...")
    company = state.get("company", "")

    # ── SQL QUERY 1: Full latest records ─────────────────────────
    q1_rows = tool_sql_query(
        "SELECT id, run_ts, company, ticker, price, currency, market_cap, "
        "pe_ratio, week52_high, week52_low, volume, avg_volume, sector, industry "
        "FROM financial_snapshots ORDER BY run_ts DESC LIMIT 5"
    )
    print(f"   ✓ Query 1: {len(q1_rows)} rows retrieved")

    # ── SQL QUERY 2: 52-week position % ──────────────────────────
    q2_rows = tool_sql_query(
        "SELECT company, ticker, price, week52_low, week52_high, "
        "CASE WHEN (week52_high - week52_low) > 0 "
        "THEN ROUND(CAST(price - week52_low AS FLOAT) / (week52_high - week52_low) * 100, 1) "
        "ELSE NULL END AS position_in_52w_range_pct "
        "FROM financial_snapshots WHERE price IS NOT NULL ORDER BY run_ts DESC LIMIT 5"
    )
    print(f"   ✓ Query 2: 52-week position analysis done")

    # ── SQL QUERY 3: Volume vs average ───────────────────────────
    q3_rows = tool_sql_query(
        "SELECT company, ticker, volume, avg_volume, "
        "CASE WHEN avg_volume > 0 "
        "THEN ROUND(CAST(volume AS FLOAT) / avg_volume * 100, 1) "
        "ELSE NULL END AS volume_vs_avg_pct "
        "FROM financial_snapshots WHERE volume IS NOT NULL ORDER BY run_ts DESC LIMIT 5"
    )
    print(f"   ✓ Query 3: Volume analysis done")

    def rows_to_text(rows: list[dict]) -> str:
        if not rows: return "  (no data)\n"
        lines = []
        for r in rows:
            lines.append("  " + " | ".join(f"{k}: {v}" for k, v in r.items() if v is not None))
        return "\n".join(lines)

    # ── LLM: synthesise SQL findings ─────────────────────────────
    sql_context = (
        f"Query 1 (latest records):\n{rows_to_text(q1_rows)}\n\n"
        f"Query 2 (52-week position):\n{rows_to_text(q2_rows)}\n\n"
        f"Query 3 (volume vs avg):\n{rows_to_text(q3_rows)}"
    )
    analysis = llm(temp=0.2).invoke([
        SystemMessage(content="You are a quantitative analyst interpreting SQL query results. Be precise."),
        HumanMessage(content=(
            f"Company: {company}\n\nSQL Results:\n{sql_context}\n\n"
            "Provide:\n"
            "1. THREE distinct data-driven observations from these SQL results\n"
            "2. Any notable anomalies or signals\n"
            "3. One-sentence DATA VERDICT summarising what the numbers say overall\n"
            "Be concise and number-specific."
        )),
    ]).content

    intel = (
        "╔═══════════════════════════════════════════╗\n"
        "║  🗄️  SQL INTELLIGENCE CHAIN  (Agent 5)   ║\n"
        "║  Chain: Agent 3 SQL write → Agent 5 read ║\n"
        "╚═══════════════════════════════════════════╝\n\n"
        f"SQL Query 1 — Latest Records:\n{rows_to_text(q1_rows)}\n\n"
        f"SQL Query 2 — 52-Week Range Position:\n{rows_to_text(q2_rows)}\n\n"
        f"SQL Query 3 — Volume vs Average:\n{rows_to_text(q3_rows)}\n\n"
        f"DATA INTELLIGENCE ANALYSIS:\n{analysis}"
    )
    return {
        **state,
        "sql_intelligence": intel,
        "logs": [f"🗄️ Agent 5 [SQL Intel]: 3 queries run on DB → intelligence extracted"],
    }


# ─── AGENT 6 ─── Master Synthesiser  (SEQUENTIAL) ────────────────────────────

def agent_master_synthesiser(state: IntelState) -> dict:
    """
    AGENT 6 — Master Synthesiser
    ══════════════════════════════
    Tool  : Wikipedia REST API (additional context)
    Role  : Chains ALL prior agent outputs into one strategic report.
    Mode  : SEQUENTIAL (final node)
    Chain :
      Wikipedia (background) +
      News brief (Agent 2)   +
      Finance data (Agent 3) +
      Weather context (Agent 4) +
      SQL intelligence (Agent 5)
      → Final Strategic Intelligence Report

    ✅ May use **state — final sequential node.
    """
    print("\n🧠 [Agent 6 — Master Synthesiser] Running (final)...")
    company  = state.get("company", "")
    topic    = state.get("topic",   "")
    query    = state.get("user_query", "")

    # ── Wikipedia: background context ────────────────────────────
    wiki = tool_wikipedia(company or topic)
    print(f"   ✓ Wikipedia: context fetched for '{company or topic}'")

    parallel = state.get("parallel_outputs", [])
    news_out    = next((p for p in parallel if "NEWS INTELLIGENCE"    in p), "Not available.")
    finance_out = next((p for p in parallel if "FINANCE + SQL CHAIN"  in p), "Not available.")
    weather_out = next((p for p in parallel if "WEATHER + GEO CHAIN"  in p), "Not available.")
    sql_out     = state.get("sql_intelligence", "Not available.")
    print(f"   ✓ Chaining {len(parallel)} parallel outputs + SQL intel + Wikipedia")

    prompt = f"""You are a Chief Intelligence Officer writing a final strategic report.

QUERY: "{query}"
COMPANY: {company} | TOPIC: {topic}

You have 5 chained intelligence sources:

━━━ SOURCE 1: WIKIPEDIA BACKGROUND ━━━
{wiki}

━━━ SOURCE 2: NEWS INTELLIGENCE (RSS → DuckDuckGo chain) ━━━
{news_out[:1200]}

━━━ SOURCE 3: FINANCIAL DATA (API → SQL chain) ━━━
{finance_out[:1200]}

━━━ SOURCE 4: WEATHER + GEO CONTEXT (Geocode → Weather chain) ━━━
{weather_out[:1000]}

━━━ SOURCE 5: SQL DATABASE INTELLIGENCE ━━━
{sql_out[:1000]}

Write a STRATEGIC INTELLIGENCE REPORT with these sections:

## 🎯 EXECUTIVE SUMMARY
(3-4 sentences. The whole story in one paragraph.)

## 📊 MARKET INTELLIGENCE
(Synthesise news + financial data. Key developments, valuation context.)

## 🔢 DATA & METRICS
(Specific numbers from API + SQL. What they mean.)

## 🌦️ MACRO & CONTEXTUAL FACTORS
(Weather-to-business impact. Macro signals.)

## ⚠️ RISK ASSESSMENT
(3–5 specific risks from across all sources.)

## ✅ STRATEGIC RECOMMENDATIONS
(4–5 specific, actionable recommendations.)

## 🔗 TOOL CHAIN TRACE
(One paragraph explaining exactly which tools chained into which outputs.)

Use markdown. Be specific. Cite sources. 600–900 words."""

    report = llm(temp=0.4, tokens=2000).invoke([
        SystemMessage(content="You are a legendary investment intelligence chief. Write with authority and precision."),
        HumanMessage(content=prompt),
    ]).content
    print(f"   ✓ Final report ready ({len(report)} chars)")

    return {
        **state,
        "final_report": report,
        "logs": [f"🧠 Agent 6 [Synthesiser]: Wikipedia + {len(parallel)} parallel outputs + SQL → report done"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Builds the LangGraph StateGraph.

    FLOW:
      agent_query_router (sequential)
         │
         ├──→ agent_news_researcher  ─┐
         ├──→ agent_finance_sql      ─┤  (3-way parallel fan-out)
         └──→ agent_weather_geo      ─┘
                                      │ fan-in (all 3 done)
                                      ▼
                           agent_sql_intelligence (sequential)
                                      │
                                      ▼
                           agent_master_synthesiser (sequential)
                                      │
                                      ▼
                                     END
    """
    g = StateGraph(IntelState)

    g.add_node("router",      agent_query_router)
    g.add_node("news",        agent_news_researcher)
    g.add_node("finance_sql", agent_finance_sql)
    g.add_node("weather_geo", agent_weather_geo)
    g.add_node("sql_intel",   agent_sql_intelligence)
    g.add_node("synthesiser", agent_master_synthesiser)

    g.set_entry_point("router")

    # fan-out  (3 parallel branches)
    g.add_edge("router",      "news")
    g.add_edge("router",      "finance_sql")
    g.add_edge("router",      "weather_geo")

    # fan-in  → sql_intel waits for all 3
    g.add_edge("news",        "sql_intel")
    g.add_edge("finance_sql", "sql_intel")
    g.add_edge("weather_geo", "sql_intel")

    # sequential tail
    g.add_edge("sql_intel",   "synthesiser")
    g.add_edge("synthesiser", END)

    return g.compile()


GRAPH = build_graph()


def run_pipeline(user_query: str) -> IntelState:
    """Entry point: run the full 6-agent pipeline."""
    print(f"\n{'═'*65}")
    print(f"  🚀  ADVANCED TOOL-CHAIN PIPELINE")
    print(f"  Query : {user_query}")
    print(f"{'═'*65}")
    initial: IntelState = {
        "user_query":       user_query,
        "company":          None,
        "ticker":           None,
        "topic":            None,
        "city":             None,
        "country":          None,
        "parallel_outputs": [],
        "sql_intelligence": None,
        "final_report":     None,
        "logs":             [],
    }
    result = GRAPH.invoke(initial)
    print(f"\n{'═'*65}")
    print(f"  ✅  PIPELINE COMPLETE — {len(result.get('logs',[]))} steps logged")
    print(f"{'═'*65}\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

def run_app():
    import streamlit as st

    st.set_page_config(
        page_title="Advanced Tool-Chain Intelligence",
        page_icon="⛓️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── STYLES ────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink:   #05070f;
        --s1:    #0b1120;
        --s2:    #111b30;
        --s3:    #18263f;
        --line:  #1e3050;
        --a1:    #00e5ff;   /* cyan   – sequential */
        --a2:    #ff6b35;   /* orange – parallel 1 (news) */
        --a3:    #00e676;   /* green  – parallel 2 (finance) */
        --a4:    #ffd600;   /* yellow – parallel 3 (weather) */
        --a5:    #e040fb;   /* violet – sql intel */
        --a6:    #ff4081;   /* pink   – synthesiser */
        --txt:   #dce8ff;
        --muted: #4a6080;
    }

    html,body,.stApp {
        background: var(--ink) !important;
        color: var(--txt) !important;
        font-family: 'Syne', sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background: var(--s1) !important;
        border-right: 1px solid var(--line) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--txt) !important; }

    /* ── hero ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: clamp(1.8rem, 4vw, 3rem);
        letter-spacing: -1px;
        background: linear-gradient(120deg, var(--a1), var(--a2), var(--a3), var(--a4), var(--a5), var(--a6));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin: 0;
    }
    .hero-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* ── pipeline diagram ── */
    .pipe-row {
        display: flex; align-items: center;
        justify-content: center; gap: 6px;
        margin: 1.2rem 0; flex-wrap: wrap;
    }
    .pipe-node {
        padding: 0.5rem 0.9rem; border-radius: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem; font-weight: 500;
        border: 1px solid; text-align: center;
        line-height: 1.4;
    }
    .pipe-arrow { color: var(--muted); font-size: 1.1rem; }
    .pipe-par-wrap {
        display: flex; flex-direction: column; align-items: center; gap: 4px;
    }
    .pipe-par-badge {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
        color: var(--muted); padding: 1px 8px;
    }
    .pipe-par-nodes { display: flex; gap: 6px; align-items: center; }

    /* ── agent sidebar cards ── */
    .ag-wrap { margin-bottom: 0.5rem; }
    .ag-card {
        background: var(--s2);
        border: 1px solid var(--line);
        border-radius: 9px;
        padding: 0.75rem 0.9rem;
    }
    .ag-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
    .ag-name { font-family: 'IBM Plex Mono', monospace; font-weight: 500; font-size: 0.78rem; }
    .ag-mode {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
        padding: 2px 7px; border-radius: 20px; border: 1px solid;
    }
    .ag-chain { font-size: 0.7rem; color: var(--muted); line-height: 1.5; }

    /* ── log ── */
    .log-line {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem; color: var(--a3);
        padding: 0.25rem 0;
        border-bottom: 1px solid var(--line);
    }

    /* ── example pill buttons ── */
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--line) !important;
        color: var(--txt) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important;
        border-radius: 6px !important;
        padding: 0.35rem 0.5rem !important;
    }
    .stButton > button:hover {
        border-color: var(--a1) !important;
        color: var(--a1) !important;
    }
    .run-btn > button {
        background: linear-gradient(135deg, #00e5ff22, #ff408122) !important;
        border: 1px solid var(--a1) !important;
        color: var(--a1) !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
        padding: 0.7rem !important;
        width: 100% !important;
    }

    /* ── inputs ── */
    .stTextInput > div > div > input {
        background: var(--s2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        color: var(--txt) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stSelectbox > div { background: var(--s2) !important; }

    /* ── tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        color: var(--muted) !important;
    }
    .stTabs [aria-selected="true"] { color: var(--a1) !important; }
    .stTextArea textarea {
        background: var(--s2) !important;
        border: 1px solid var(--line) !important;
        color: var(--txt) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:1.2rem 0 0.8rem;'>
            <div style='font-size:2.2rem;'>⛓️</div>
            <div style='font-family:Syne,sans-serif;font-weight:800;font-size:1rem;
                        background:linear-gradient(135deg,#00e5ff,#ff4081);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;'>Advanced Tool-Chain</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;
                        color:#4a6080;margin-top:2px;'>LangGraph · Groq · 7 Tools · 6 Agents</div>
        </div>
        """, unsafe_allow_html=True)

        groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        st.markdown("---")

        agents_cfg = [
            ("--a1", "sequential", "🧭", "Agent 1", "Query Router",
             "🔧 LLM Parser\n→ extracts entities"),
            ("--a2", "parallel",   "📰", "Agent 2", "News Researcher",
             "🔧 Google RSS\n→ 🔧 DuckDuckGo"),
            ("--a3", "parallel",   "📈", "Agent 3", "Finance + SQL",
             "🔧 yFinance API\n→ 🔧 SQLite Write"),
            ("--a4", "parallel",   "🌦️", "Agent 4", "Weather + Geo",
             "🔧 Geocoding API\n→ 🔧 Weather API"),
            ("--a5", "sequential", "🗄️", "Agent 5", "SQL Intelligence",
             "🔧 SQLite Read\n← chains Agent 3 DB"),
            ("--a6", "sequential", "🧠", "Agent 6", "Master Synthesiser",
             "🔧 Wikipedia\n→ LLM Report"),
        ]
        for var, mode, icon, num, name, chain in agents_cfg:
            bc = "#00e5ff" if mode == "sequential" else "#ff6b35"
            st.markdown(f"""
            <div class='ag-wrap'>
              <div class='ag-card' style='border-left:3px solid var({var});'>
                <div class='ag-head'>
                  <span style='font-size:1rem;'>{icon}</span>
                  <span class='ag-name' style='color:var({var});'>{num} — {name}</span>
                  <span class='ag-mode' style='border-color:{bc}44;color:{bc};background:{bc}11;'>{mode}</span>
                </div>
                <div class='ag-chain' style='white-space:pre-line;'>{chain}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#4a6080;line-height:1.8;'>
            Free Groq key:<br>console.groq.com<br><br>
            Optional (real stock data):<br>pip install yfinance<br><br>
            DB: intelligence.db (SQLite)
        </div>
        """, unsafe_allow_html=True)

    # ── MAIN ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem;'>
        <p class='hero-title'>Advanced Tool-Chain<br>Intelligence System</p>
        <p class='hero-sub'>
            ⛓️ LANGGRAPH · GROQ llama-3.3-70b · 6 AGENTS · 7 TOOLS · SEQUENTIAL + PARALLEL
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("""
    <div class='pipe-row'>
        <div class='pipe-node' style='border-color:#00e5ff;color:#00e5ff;background:#00e5ff11;'>
            🧭 Agent 1<br>Query Router<br><span style='font-size:0.58rem;color:#4a6080;'>LLM Parser</span>
        </div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-par-wrap'>
            <div class='pipe-par-badge'>── parallel fan-out ──</div>
            <div class='pipe-par-nodes'>
                <div class='pipe-node' style='border-color:#ff6b35;color:#ff6b35;background:#ff6b3511;'>
                    📰 Agent 2<br>News<br><span style='font-size:0.58rem;color:#4a6080;'>RSS→DDG</span>
                </div>
                <div style='color:#4a6080;'>⚡</div>
                <div class='pipe-node' style='border-color:#00e676;color:#00e676;background:#00e67611;'>
                    📈 Agent 3<br>Finance+SQL<br><span style='font-size:0.58rem;color:#4a6080;'>API→SQLite</span>
                </div>
                <div style='color:#4a6080;'>⚡</div>
                <div class='pipe-node' style='border-color:#ffd600;color:#ffd600;background:#ffd60011;'>
                    🌦️ Agent 4<br>Weather+Geo<br><span style='font-size:0.58rem;color:#4a6080;'>Geo→Wx</span>
                </div>
            </div>
            <div class='pipe-par-badge'>── fan-in (all 3 complete) ──</div>
        </div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-node' style='border-color:#e040fb;color:#e040fb;background:#e040fb11;'>
            🗄️ Agent 5<br>SQL Intel<br><span style='font-size:0.58rem;color:#4a6080;'>3 SQL Queries</span>
        </div>
        <div class='pipe-arrow'>→</div>
        <div class='pipe-node' style='border-color:#ff4081;color:#ff4081;background:#ff408111;'>
            🧠 Agent 6<br>Synthesiser<br><span style='font-size:0.58rem;color:#4a6080;'>Wiki→Report</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Examples
    examples = [
        "Analyse Tesla and the weather in Austin Texas",
        "Research Apple Inc stock and news",
        "Tell me about Infosys and weather in Bangalore",
        "Analyse Reliance Industries, Mumbai",
        "Research NVIDIA and the weather in San Jose",
        "Tell me about Microsoft Azure cloud business",
    ]
    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(ex[:22] + "…", key=f"ex_{i}", use_container_width=True):
                st.session_state["pf"] = ex

    pf = st.session_state.pop("pf", "")
    query = st.text_input(
        "Query",
        value=pf,
        placeholder="e.g. 'Analyse Tesla and weather in Austin' — include a company + city for best results",
        label_visibility="collapsed",
    )

    st.markdown("<div class='run-btn'>", unsafe_allow_html=True)
    run_btn = st.button("⛓️ Run Advanced Tool-Chain Pipeline", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        if not os.environ.get("GROQ_API_KEY"):
            st.warning("⚠️ Enter your Groq API key in the sidebar. Free at console.groq.com")
        elif not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.status("⛓️ Pipeline running — 6 agents, 7 tools…", expanded=True) as status:
                st.write("🧭 **Agent 1** — Parsing query with LLM…")
                st.write("📰 **Agent 2** — Google RSS → DuckDuckGo *(parallel)*")
                st.write("📈 **Agent 3** — yFinance API → SQLite write *(parallel)*")
                st.write("🌦️ **Agent 4** — Geocoding → Weather API *(parallel)*")
                st.write("🗄️ **Agent 5** — Running 3 SQL queries on DB…")
                st.write("🧠 **Agent 6** — Wikipedia → final synthesis…")
                try:
                    result = run_pipeline(query.strip())
                    status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

                    # Entity tags
                    c1, c2, c3, c4, c5 = st.columns(5)
                    for col, label, val, color in [
                        (c1, "Company",  result.get("company",  ""), "#00e5ff"),
                        (c2, "Ticker",   result.get("ticker",   ""), "#00e676"),
                        (c3, "Topic",    result.get("topic",    ""), "#ff6b35"),
                        (c4, "City",     result.get("city",     ""), "#ffd600"),
                        (c5, "Country",  result.get("country",  ""), "#e040fb"),
                    ]:
                        with col:
                            st.markdown(
                                f"<div style='background:{color}11;border:1px solid {color}44;"
                                f"border-radius:8px;padding:0.5rem 0.75rem;"
                                f"font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
                                f"color:{color};'><b>{label}</b><br>{val}</div>",
                                unsafe_allow_html=True
                            )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Execution log
                    st.markdown("#### 📋 Execution Log")
                    log_html = "".join(
                        f"<div class='log-line'>{l}</div>"
                        for l in result.get("logs", [])
                    )
                    st.markdown(
                        f"<div style='background:var(--s1);border:1px solid var(--line);"
                        f"border-radius:10px;padding:0.8rem 1rem;'>{log_html}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Output tabs
                    tabs = st.tabs([
                        "🧠 Final Report",
                        "📰 News Intel",
                        "📈 Finance + SQL",
                        "🌦️ Weather + Geo",
                        "🗄️ SQL Intelligence",
                    ])
                    parallel = result.get("parallel_outputs", [])
                    news_out    = next((p for p in parallel if "NEWS INTELLIGENCE"    in p), "")
                    finance_out = next((p for p in parallel if "FINANCE + SQL CHAIN"  in p), "")
                    weather_out = next((p for p in parallel if "WEATHER + GEO CHAIN"  in p), "")

                    with tabs[0]:
                        st.markdown(result.get("final_report", "*Not generated.*"))
                    with tabs[1]:
                        st.text_area("", value=news_out,    height=500, label_visibility="collapsed")
                    with tabs[2]:
                        st.text_area("", value=finance_out, height=500, label_visibility="collapsed")
                    with tabs[3]:
                        st.text_area("", value=weather_out, height=500, label_visibility="collapsed")
                    with tabs[4]:
                        st.text_area("", value=result.get("sql_intelligence", ""), height=500, label_visibility="collapsed")

                except Exception as e:
                    status.update(label="❌ Pipeline Failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.info("Check Groq API key · Internet connection · pip install -r requirements.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_app()
