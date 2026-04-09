"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           DAILY BRIEFING CREW — main.py                                    ║
║                                                                              ║
║  Entry point for the Daily Briefing CrewAI application.                    ║
║  Supports two run modes:                                                    ║
║    1. CLI   : python main.py                                                ║
║    2. UI    : streamlit run main.py                                         ║
║    3. CrewAI: uv run crewai run  (uses the run() function below)           ║
║                                                                              ║
║  Sequential Flow:                                                            ║
║  IPL Agent → AI News Agent → Weather Agent → Movies Agent → Supervisor     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure output directory exists
Path("output").mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# run() — REQUIRED BY CrewAI  (uv run crewai run calls this)
# ══════════════════════════════════════════════════════════════════════════════

def run():
    """
    Required entry point for CrewAI CLI.
    Called automatically when you run:  uv run crewai run
    """
    from crew import DailyBriefingCrew

    inputs = {
        "cities":     "Mumbai, Delhi, London",
        "movie_city": "Mumbai",
        "date":       datetime.now().strftime("%A, %d %B %Y"),
    }

    print("\n" + "=" * 65)
    print("  DAILY BRIEFING CREW — STARTING")
    print(f"  Weather Cities : {inputs['cities']}")
    print(f"  Movie City     : {inputs['movie_city']}")
    print(f"  Date           : {inputs['date']}")
    print("=" * 65 + "\n")

    DailyBriefingCrew().crew().kickoff(inputs=inputs)


# ══════════════════════════════════════════════════════════════════════════════
# train() — REQUIRED BY CrewAI  (uv run crewai train calls this)
# ══════════════════════════════════════════════════════════════════════════════

def train():
    """
    Required entry point for CrewAI CLI training.
    Called automatically when you run:  uv run crewai train
    """
    from crew import DailyBriefingCrew

    inputs = {
        "cities":     "Mumbai, Delhi, London",
        "movie_city": "Mumbai",
        "date":       datetime.now().strftime("%A, %d %B %Y"),
    }

    try:
        n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        filename     = sys.argv[3] if len(sys.argv) > 3 else "training_output.pkl"
        DailyBriefingCrew().crew().train(
            n_iterations=n_iterations,
            filename=filename,
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"Training failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# replay() — REQUIRED BY CrewAI  (uv run crewai replay calls this)
# ══════════════════════════════════════════════════════════════════════════════

def replay():
    """
    Required entry point for CrewAI CLI replay.
    Called automatically when you run:  uv run crewai replay
    """
    from crew import DailyBriefingCrew

    try:
        task_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not task_id:
            raise ValueError("Please provide a task_id: uv run crewai replay <task_id>")
        DailyBriefingCrew().crew().replay(task_id=task_id)
    except Exception as e:
        raise Exception(f"Replay failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# test() — REQUIRED BY CrewAI  (uv run crewai test calls this)
# ══════════════════════════════════════════════════════════════════════════════

def test():
    """
    Required entry point for CrewAI CLI testing.
    Called automatically when you run:  uv run crewai test
    """
    from crew import DailyBriefingCrew

    inputs = {
        "cities":     "Mumbai, Delhi, London",
        "movie_city": "Mumbai",
        "date":       datetime.now().strftime("%A, %d %B %Y"),
    }

    try:
        n_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        openai_model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"
        DailyBriefingCrew().crew().test(
            n_iterations=n_iterations,
            openai_model_name=openai_model,
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"Test failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUN FUNCTION  (used by CLI and Streamlit modes)
# ══════════════════════════════════════════════════════════════════════════════

def run_crew(
    cities: list[str],
    movie_city: str,
    groq_api_key: str = None,
) -> dict:
    """
    Runs the DailyBriefingCrew with the given inputs.

    Args:
        cities       : List of city names for weather (e.g. ["Mumbai", "Delhi"])
        movie_city   : City name for movie listings (e.g. "Mumbai")
        groq_api_key : Optional — Groq API key (overrides .env if provided)

    Returns:
        dict with keys:
          - "result"   : Final briefing text (string)
          - "success"  : True/False
          - "error"    : Error message if success=False
    """
    # Set Groq API key
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    if not os.environ.get("GROQ_API_KEY"):
        return {
            "result": "",
            "success": False,
            "error": "GROQ_API_KEY not found. Set it in .env or provide it directly."
        }

    # Import here to avoid loading before API key is set
    from crew import DailyBriefingCrew

    # Prepare inputs — these replace {placeholders} in tasks.yaml
    inputs = {
        "cities":     ", ".join(cities),
        "movie_city": movie_city,
        "date":       datetime.now().strftime("%A, %d %B %Y"),
    }

    print("\n" + "=" * 65)
    print("  DAILY BRIEFING CREW — STARTING")
    print(f"  Weather Cities  : {inputs['cities']}")
    print(f"  Movie City      : {movie_city}")
    print(f"  Date            : {inputs['date']}")
    print("=" * 65 + "\n")

    try:
        result     = DailyBriefingCrew().crew().kickoff(inputs=inputs)
        final_text = result.raw if hasattr(result, "raw") else str(result)

        print("\n" + "=" * 65)
        print("  CREW COMPLETED SUCCESSFULLY")
        print("  Output saved to: output/daily_briefing.md")
        print("=" * 65 + "\n")

        return {"result": final_text, "success": True, "error": None}

    except Exception as e:
        error_msg = f"Crew run failed: {str(e)}"
        print(f"\n ERROR: {error_msg}\n")
        return {"result": "", "success": False, "error": error_msg}


# ══════════════════════════════════════════════════════════════════════════════
# CLI MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_cli():
    """
    Command-line interface mode.
    Run with: python main.py
    """
    print("\n Daily Briefing Crew — CLI Mode")
    print("-" * 40)

    # Defaults — modify these for quick testing
    cities     = ["Mumbai", "Delhi", "London", "New York", "Tokyo"]
    movie_city = "Mumbai"

    output = run_crew(cities=cities, movie_city=movie_city)

    if output["success"]:
        print("\n" + "=" * 65)
        print("FINAL DAILY BRIEFING")
        print("=" * 65)
        print(output["result"])
    else:
        print(f"\n ERROR: {output['error']}")


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_app():
    """
    Streamlit web UI mode.
    Run with: streamlit run main.py
    """
    import streamlit as st

    # ── Page config ────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Daily Briefing Crew",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:       #0a0f1e;
        --surface:  #111827;
        --surface2: #1a2235;
        --border:   #1e3a5f;
        --accent1:  #f59e0b;
        --accent2:  #38bdf8;
        --accent3:  #34d399;
        --accent4:  #f472b6;
        --accent5:  #a78bfa;
        --text:     #e2e8f0;
        --muted:    #64748b;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .hero h1 {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #f59e0b, #38bdf8, #34d399, #f472b6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }

    .flow-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .flow-chip {
        padding: 0.4rem 0.85rem;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        border: 1px solid;
    }
    .flow-arrow { color: var(--muted); font-size: 1rem; padding: 0 2px; }

    .agent-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .agent-name { font-weight: 600; font-size: 0.82rem; margin-bottom: 0.15rem; }
    .agent-desc { font-size: 0.72rem; color: var(--muted); line-height: 1.4; }

    .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #38bdf8) !important;
        color: #0a0f1e !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100% !important;
        padding: 0.7rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }
    .stMultiSelect > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .briefing-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.8;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 0.5rem;'>
            <div style='font-size:2.5rem;'>📰</div>
            <div style='font-weight:800; font-size:1.1rem; color:#f59e0b;'>Daily Briefing Crew</div>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem; color:#64748b;'>
                CrewAI + Groq — Sequential Agents
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # API Key
        groq_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com"
        )

        st.markdown("---")

        # Agent pipeline legend
        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.65rem;
                    color:#64748b; text-transform:uppercase; letter-spacing:0.8px;
                    margin-bottom:0.5rem;'>Agent Pipeline</div>
        """, unsafe_allow_html=True)

        agents_info = [
            ("#f59e0b", "🏏", "IPL Agent",       "DuckDuckGo → IPL 2026 news"),
            ("#38bdf8", "🤖", "AI News Agent",    "DuckDuckGo → AI developments"),
            ("#34d399", "🌤️", "Weather Agent",    "Open-Meteo → Live weather"),
            ("#f472b6", "🎬", "Movies Agent",     "DuckDuckGo → Theatre listings"),
            ("#a78bfa", "🧠", "Supervisor Agent", "LLM → Compile final briefing"),
        ]
        for color, icon, name, desc in agents_info:
            st.markdown(f"""
            <div class='agent-card' style='border-left:3px solid {color};'>
                <div class='agent-name' style='color:{color};'>{icon} {name}</div>
                <div class='agent-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; color:#f59e0b;'>
            🆓 Free Groq key:<br>console.groq.com<br><br>
            📦 Model: llama-3.3-70b-versatile
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ───────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero'>
        <h1>📰 Daily Briefing Crew</h1>
        <p>⚡ CrewAI · Groq · Sequential Agents · IPL · AI News · Weather · Movies</p>
    </div>
    """, unsafe_allow_html=True)

    # Sequential flow diagram
    st.markdown("""
    <div class='flow-row'>
        <div class='flow-chip' style='border-color:#f59e0b;color:#f59e0b;background:#f59e0b18;'>🏏 IPL Agent</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-chip' style='border-color:#38bdf8;color:#38bdf8;background:#38bdf818;'>🤖 AI News Agent</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-chip' style='border-color:#34d399;color:#34d399;background:#34d39918;'>🌤️ Weather Agent</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-chip' style='border-color:#f472b6;color:#f472b6;background:#f472b618;'>🎬 Movies Agent</div>
        <div class='flow-arrow'>→</div>
        <div class='flow-chip' style='border-color:#a78bfa;color:#a78bfa;background:#a78bfa18;'>🧠 Supervisor Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Inputs ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌤️ Weather Cities")
        city_options = [
            "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
            "Kolkata", "Pune", "Jaipur", "Ahmedabad",
            "London", "New York", "Tokyo", "Dubai",
            "Singapore", "Paris", "Sydney", "Toronto"
        ]
        selected_cities = st.multiselect(
            "Select cities for weather report",
            options=city_options,
            default=["Mumbai", "Delhi", "London"],
            label_visibility="collapsed",
        )
        custom = st.text_input(
            "Add a custom city",
            placeholder="e.g. Surat, Zurich, Cape Town"
        )
        if custom.strip():
            selected_cities.append(custom.strip())

    with col2:
        st.markdown("#### 🎬 Movie City")
        movie_city = st.text_input(
            "City for theatre listings",
            value="Mumbai",
            placeholder="e.g. Mumbai, Bangalore, Delhi"
        )

        st.markdown("#### 📅 Today's Date")
        st.info(f"🗓️ {datetime.now().strftime('%A, %d %B %Y')}")

        st.markdown("""
        <div style='background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                    padding:0.6rem 0.85rem; font-family:JetBrains Mono,monospace;
                    font-size:0.7rem; color:#64748b; margin-top:0.5rem;'>
            ⏱️ Runtime: ~90-120 seconds<br>
            5 agents run sequentially via CrewAI
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    run_btn = st.button("🚀 Generate Daily Briefing", use_container_width=True)

    # ── Run ────────────────────────────────────────────────────────────────
    if run_btn:
        if not groq_key and not os.environ.get("GROQ_API_KEY"):
            st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        elif not selected_cities:
            st.warning("⚠️ Please select at least one city for weather.")
        elif not movie_city.strip():
            st.warning("⚠️ Please enter a city for movie listings.")
        else:
            with st.spinner("🚀 Running 5-agent CrewAI pipeline... (this takes ~90-120 seconds)"):
                output = run_crew(
                    cities=selected_cities,
                    movie_city=movie_city.strip(),
                    groq_api_key=groq_key or None,
                )

            if output["success"]:
                st.success("✅ Daily Briefing Generated Successfully!")

                st.markdown("### 📰 Your Daily Briefing")
                st.markdown(
                    f"<div class='briefing-box'>{output['result'].replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )

                # Download button
                st.download_button(
                    label="⬇️ Download Briefing (.md)",
                    data=output["result"],
                    file_name=f"daily_briefing_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                )
            else:
                st.error(f"❌ Error: {output['error']}")
                st.markdown("""
                **Troubleshooting tips:**
                - Check your Groq API key is valid at console.groq.com
                - Ensure all packages are installed: `pip install -r requirements.txt`
                - Check your internet connection (DuckDuckGo and Open-Meteo need network access)
                """)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Detect if running via Streamlit or as plain Python
    try:
        import streamlit as st
        if any("streamlit" in arg for arg in sys.argv):
            run_app()
        else:
            run_cli()
    except Exception:
        run_cli()
