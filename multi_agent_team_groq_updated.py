"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              MULTI-AGENT TEAM SYSTEM — Powered by Groq API                 ║
║                                                                              ║
║  Agents:                                                                     ║
║   1. SupervisorAgent   — Orchestrates tasks, routes to sub-agents           ║
║   2. ResearchAgent     — Researches a topic using Groq LLM                  ║
║   3. SummarizerAgent   — Summarizes long content into concise output        ║
║   4. EmailAgent        — Drafts AND sends emails via Gmail SMTP             ║
║                                                                              ║
║  Requirements:                                                               ║
║   pip install groq python-dotenv                                            ║
║                                                                              ║
║  EMAIL SETUP (Gmail):                                                        ║
║   1. Enable 2-Step Verification on your Google Account                      ║
║   2. Go to: myaccount.google.com → Security → App Passwords                ║
║   3. Generate an App Password for "Mail"                                    ║
║   4. Copy the 16-char password into your .env as SMTP_PASSWORD              ║
║                                                                              ║
║  .env file template:                                                         ║
║   GROQ_API_KEY=gsk_your_key_here                                            ║
║   SMTP_HOST=smtp.gmail.com                                                  ║
║   SMTP_PORT=587                                                              ║
║   SMTP_USER=yourname@gmail.com                                              ║
║   SMTP_PASSWORD=abcd efgh ijkl mnop   ← Gmail App Password (16 chars)      ║
║                                                                              ║
║  Groq Models Available:                                                      ║
║   - llama-3.3-70b-versatile     (default, best quality)                    ║
║   - llama-3.1-8b-instant        (fastest, lightweight)                     ║
║   - mixtral-8x7b-32768          (large context window)                     ║
║   - gemma2-9b-it                (Google Gemma 2)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import re
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "<your-groq-api-key>")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")

# ── Gmail SMTP settings ───────────────────────────────────────────────────────
# FIX: Changed default host from smtp.office365.com → smtp.gmail.com
# Set these in your .env file — see header above for instructions.
SMTP_HOST     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER",     "")        # your Gmail address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")        # Gmail App Password (16 chars)

# ── Available Groq models for easy switching ──────────────────────────────────
GROQ_MODELS = {
    "fast"    : "llama-3.1-8b-instant",
    "balanced": "llama-3.3-70b-versatile",
    "large"   : "mixtral-8x7b-32768",
    "gemma"   : "gemma2-9b-it",
}

# ─────────────────────────────────────────────
# 1. GROQ CLIENT
# ─────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


def call_groq(
    system_prompt : str,
    user_message  : str,
    temperature   : float = 0.7,
    model         : str   = None,
    max_tokens    : int   = 2048,
) -> str:
    """
    Central Groq LLM caller.
    Falls back to the global GROQ_MODEL if none is specified.
    """
    model = model or GROQ_MODEL
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 2. BASE AGENT CLASS
# ─────────────────────────────────────────────
class BaseAgent:
    """Shared foundation for every agent."""

    def __init__(self, name: str, role: str, instructions: str, model: str = None):
        self.name         = name
        self.role         = role
        self.instructions = instructions
        self.model        = model or GROQ_MODEL
        self.memory: list[dict] = []

    def run(self, task: str) -> str:
        raise NotImplementedError("Each agent must implement run()")

    def _log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] 🤖 [{self.name}] {message}")

    def _remember(self, task: str, result: str):
        self.memory.append({
            "task"     : task,
            "result"   : result,
            "timestamp": str(datetime.now()),
            "model"    : self.model,
        })

    def _call(self, user_message: str, temperature: float = 0.7) -> str:
        """Convenience wrapper using this agent's own model."""
        return call_groq(
            system_prompt=self.instructions,
            user_message=user_message,
            temperature=temperature,
            model=self.model,
        )


# ─────────────────────────────────────────────
# 3. RESEARCH AGENT
# ─────────────────────────────────────────────
class ResearchAgent(BaseAgent):
    """
    Researches any topic and returns a structured report.
    Uses Groq's 70B model for high quality research output.
    """

    SYSTEM_PROMPT = """
You are ResearchBot, a highly capable AI research assistant powered by Groq.
When given a topic, you produce a structured research report with:

[OVERVIEW]
A concise background on the topic.

[KEY FACTS]
5-7 important facts, statistics, or data points.

[RECENT TRENDS]
3-5 current developments or emerging patterns.

[INSIGHTS]
2-3 expert-level insights or implications.

[SOURCES TO EXPLORE]
3-5 recommended sources, journals, or websites.

Be thorough, factual, well-organised, and professional.
    """

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            role="Researcher",
            instructions=self.SYSTEM_PROMPT,
            model=GROQ_MODELS["balanced"],   # 70B for research quality
        )

    def run(self, topic: str) -> str:
        self._log(f"Researching: '{topic}'")
        result = self._call(
            user_message=f"Please research the following topic in detail:\n\n{topic}",
            temperature=0.4,
        )
        self._remember(topic, result)
        self._log("Research complete ✓")
        return result


# ─────────────────────────────────────────────
# 4. SUMMARIZER AGENT
# ─────────────────────────────────────────────
class SummarizerAgent(BaseAgent):
    """
    Condenses long content into clean, concise summaries.
    Uses Groq's instant model for speed since summarisation is simpler.
    """

    SYSTEM_PROMPT = """
You are SummaryBot, an expert at distilling information clearly and concisely.
When given content, produce a summary in this exact format:

[KEY TAKEAWAYS]
• Point 1
• Point 2
• Point 3
• Point 4
• Point 5

[ONE-PARAGRAPH SUMMARY]
A single, clear paragraph summarising the essence.

[ACTION ITEMS]
• Action 1 (if applicable)
• Action 2 (if applicable)

Keep the language simple, professional, and to the point.
    """

    def __init__(self):
        super().__init__(
            name="SummarizerAgent",
            role="Summarizer",
            instructions=self.SYSTEM_PROMPT,
            model=GROQ_MODELS["fast"],       # 8B instant for speed
        )

    def run(self, content: str, style: str = "executive") -> str:
        self._log(f"Summarizing ({len(content)} chars, style={style})")
        result = self._call(
            user_message=(
                f"Summarize the following content in a '{style}' style:\n\n{content}"
            ),
            temperature=0.3,
        )
        self._remember(content[:80] + "...", result)
        self._log("Summary complete ✓")
        return result


# ─────────────────────────────────────────────
# 5. EMAIL AGENT  ← FIXED
# ─────────────────────────────────────────────
class EmailAgent(BaseAgent):
    """
    Drafts a professional email using Groq LLM, then SENDS it via Gmail SMTP.

    FIX SUMMARY (what was broken and what was changed):
    ─────────────────────────────────────────────────────
    1. SMTP host default was smtp.office365.com → changed to smtp.gmail.com
    2. Missing server.ehlo() calls around starttls() → added (Gmail requires this)
    3. Added specific SMTPAuthenticationError catch with a helpful message
    4. Added clear diagnostic messages so you can see exactly what's happening
    5. Strips spaces from App Password (users sometimes copy it with spaces)

    HOW TO CONFIGURE:
    ─────────────────
    Add these to your .env file:
        SMTP_HOST=smtp.gmail.com
        SMTP_PORT=587
        SMTP_USER=yourname@gmail.com
        SMTP_PASSWORD=abcd efgh ijkl mnop    ← Gmail App Password, NOT your real password

    To get a Gmail App Password:
        1. Go to myaccount.google.com
        2. Security → enable 2-Step Verification
        3. Search "App Passwords" → Generate → select Mail
        4. Copy the 16-character password shown
    """

    SYSTEM_PROMPT = """
You are MailBot, an expert email communication assistant powered by Groq.
When given content and recipient details, you draft a professional email.

Always output in this EXACT format:

Subject: <a compelling subject line>

Body:
Dear [Recipient Name / Team],

<opening line that sets the context>

<main content — well structured, clear paragraphs>

<closing line with call to action if needed>

Best regards,
<Sender Name>

Keep the tone warm, clear, professional, and action-oriented.
    """

    def __init__(self):
        super().__init__(
            name="EmailAgent",
            role="Email Dispatcher",
            instructions=self.SYSTEM_PROMPT,
            model=GROQ_MODELS["balanced"],
        )

    def run(
        self,
        subject     : str,
        content     : str,
        recipients  : list[str],
        sender_name : str = "AI Research Team",
    ) -> str:
        self._log(f"Drafting email | To: {recipients} | Subject hint: '{subject}'")

        draft_request = (
            f"Draft a professional email with the following details:\n\n"
            f"Subject hint : {subject}\n"
            f"Recipients   : {', '.join(recipients)}\n"
            f"Sender name  : {sender_name}\n\n"
            f"Content to include in the email:\n\n{content}"
        )

        drafted_email = self._call(user_message=draft_request, temperature=0.6)
        self._log("Email drafted ✓")
        self._remember(subject, drafted_email)

        # ── Print the drafted email ──────────────────────────────────────
        print("\n" + "─" * 60)
        print("📧 DRAFTED EMAIL")
        print("─" * 60)
        print(drafted_email)
        print("─" * 60 + "\n")

        # ── Send via SMTP if credentials are configured ──────────────────
        if SMTP_USER and SMTP_PASSWORD:
            self._send_email(drafted_email, recipients)
        else:
            self._log(
                "⚠  SMTP credentials not set — email drafted only, NOT sent.\n"
                "   To enable sending, add to your .env file:\n"
                "     SMTP_HOST=smtp.gmail.com\n"
                "     SMTP_PORT=587\n"
                "     SMTP_USER=yourname@gmail.com\n"
                "     SMTP_PASSWORD=<your 16-char Gmail App Password>"
            )

        return drafted_email

    # ── FIXED _send_email ─────────────────────────────────────────────────
    def _send_email(self, drafted_email: str, recipients: list[str]):
        """
        Send the drafted email via Gmail SMTP (TLS on port 587).

        KEY FIXES vs original:
          • Added server.ehlo() before AND after starttls()  ← Gmail requires this
          • Strip spaces from App Password (copied with spaces by mistake)
          • Specific SMTPAuthenticationError catch with helpful message
          • Specific timeout so it doesn't hang forever
        """
        # ── Parse subject from drafted email ─────────────────────────────
        lines = drafted_email.splitlines()
        subject = next(
            (l.replace("Subject:", "").strip()
             for l in lines if l.lower().startswith("subject:")),
            "AI Research Update",
        )

        # ── Parse body (everything after "Body:" line) ────────────────────
        body_start = next(
            (i for i, l in enumerate(lines) if l.strip().lower() == "body:"),
            1,
        )
        body = "\n".join(lines[body_start + 1:]).strip()

        # ── Build MIME message ────────────────────────────────────────────
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ", ".join(recipients)
        msg.attach(MIMEText(body, "plain"))

        # ── Strip spaces from App Password (common copy-paste issue) ──────
        clean_password = SMTP_PASSWORD.replace(" ", "")

        self._log(f"Connecting to {SMTP_HOST}:{SMTP_PORT} ...")

        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.ehlo()          # FIX: identify ourselves to the server
                server.starttls()      # upgrade the connection to TLS
                server.ehlo()          # FIX: re-identify after TLS upgrade (Gmail needs this)
                server.login(SMTP_USER, clean_password)
                server.sendmail(SMTP_USER, recipients, msg.as_string())

            self._log(f"✅ Email sent successfully to: {recipients}")

        except smtplib.SMTPAuthenticationError:
            self._log(
                "❌ Gmail authentication failed.\n"
                "   MOST LIKELY CAUSE: You used your regular Gmail password.\n"
                "   FIX: Use a Gmail App Password instead (16 characters).\n"
                "   How to get one:\n"
                "     1. Go to myaccount.google.com\n"
                "     2. Security → enable 2-Step Verification\n"
                "     3. Search 'App Passwords' → Generate → select Mail\n"
                "     4. Copy the 16-char code into SMTP_PASSWORD in your .env"
            )

        except smtplib.SMTPConnectError:
            self._log(
                f"❌ Could not connect to {SMTP_HOST}:{SMTP_PORT}.\n"
                "   Check your internet connection and SMTP_HOST / SMTP_PORT in .env."
            )

        except smtplib.SMTPRecipientsRefused as e:
            self._log(f"❌ Recipient(s) refused by server: {e.recipients}")

        except TimeoutError:
            self._log(
                f"❌ Connection to {SMTP_HOST}:{SMTP_PORT} timed out.\n"
                "   Check your firewall — port 587 may be blocked."
            )

        except Exception as exc:
            self._log(f"❌ SMTP send failed: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────
# 6. SUPERVISOR AGENT (ORCHESTRATOR)
# ─────────────────────────────────────────────
class SupervisorAgent(BaseAgent):
    """
    OrchestraMind — the central controller.
    Uses Groq to generate a JSON execution plan, then routes tasks
    dynamically to the right sub-agents in the correct order.
    """

    SYSTEM_PROMPT = """
You are OrchestraMind, the supervisor of a multi-agent AI team running on Groq.
Your sub-agents are:
  1. ResearchAgent   — researches any topic deeply
  2. SummarizerAgent — summarizes long content concisely
  3. EmailAgent      — drafts and sends professional emails

Given a user task, output ONLY a valid JSON execution plan like this:
{
  "plan": [
    {
      "step": 1,
      "agent": "ResearchAgent",
      "task": "<specific instruction for this agent>",
      "depends_on": null
    },
    {
      "step": 2,
      "agent": "SummarizerAgent",
      "task": "<specific instruction>",
      "depends_on": 1
    },
    {
      "step": 3,
      "agent": "EmailAgent",
      "task": "<specific instruction>",
      "depends_on": 2,
      "recipients": ["email@example.com"],
      "subject": "<email subject line>"
    }
  ]
}

Rules:
- Only include steps that are genuinely needed.
- depends_on = step number whose OUTPUT this step uses as input.
- For EmailAgent always include 'recipients' and 'subject'.
- Output ONLY valid JSON. No explanation. No markdown.
    """

    def __init__(self, recipients: list[str] | None = None):
        super().__init__(
            name="SupervisorAgent",
            role="Orchestrator",
            instructions=self.SYSTEM_PROMPT,
            model=GROQ_MODELS["fast"],       # Fast model for planning
        )
        self.research_agent     = ResearchAgent()
        self.summarizer_agent   = SummarizerAgent()
        self.email_agent        = EmailAgent()
        self.default_recipients = recipients or []

    # ── Main entry point ───────────────────────────────────────────────────
    def run(self, user_task: str, recipients: list[str] | None = None) -> dict:
        recipients = recipients or self.default_recipients

        print("\n" + "═" * 60)
        self._log(f"New task received: '{user_task}'")
        print("═" * 60)

        # Step 1 — Generate execution plan
        plan = self._create_plan(user_task, recipients)

        print(f"\n📋 Execution Plan ({len(plan['plan'])} step(s)):")
        for s in plan["plan"]:
            dep = f"← uses Step {s['depends_on']}" if s.get("depends_on") else "← no dependency"
            print(f"   Step {s['step']}: {s['agent']} {dep}")
        print()

        # Step 2 — Execute each step
        results      = {}
        final_output = ""

        for step in plan.get("plan", []):
            step_num   = step["step"]
            agent_name = step["agent"]
            task_desc  = step["task"]
            depends_on = step.get("depends_on")

            # Inject previous step output if this step depends on it
            input_content = task_desc
            if depends_on and depends_on in results:
                input_content = (
                    f"{task_desc}\n\n"
                    f"--- Context from Step {depends_on} ---\n"
                    f"{results[depends_on]}"
                )

            self._log(f"▶ Executing Step {step_num} → {agent_name}")

            if agent_name == "ResearchAgent":
                output = self.research_agent.run(input_content)

            elif agent_name == "SummarizerAgent":
                output = self.summarizer_agent.run(input_content)

            elif agent_name == "EmailAgent":
                email_recipients = step.get("recipients", recipients)
                email_subject    = step.get("subject", user_task[:60])
                output = self.email_agent.run(
                    subject     = email_subject,
                    content     = input_content,
                    recipients  = email_recipients,
                    sender_name = "AI Research Team (Groq)",
                )
            else:
                output = f"⚠ Unknown agent requested: {agent_name}"

            results[step_num] = output
            final_output      = output

        self._log("All steps completed ✓")
        return {
            "task"   : user_task,
            "model"  : GROQ_MODEL,
            "plan"   : plan,
            "results": results,
            "final"  : final_output,
        }

    def _create_plan(self, user_task: str, recipients: list[str]) -> dict:
        """Ask Groq LLM to produce a JSON execution plan."""
        prompt = (
            f"User task: {user_task}\n"
            f"Email recipients (use if emailing is needed): {recipients}"
        )
        raw = call_groq(
            system_prompt = self.SYSTEM_PROMPT,
            user_message  = prompt,
            temperature   = 0.1,
            model         = self.model,
        )

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._log("⚠ Plan parsing failed — using default single-step fallback.")
            return {
                "plan": [{
                    "step"      : 1,
                    "agent"     : "ResearchAgent",
                    "task"      : user_task,
                    "depends_on": None,
                }]
            }


# ─────────────────────────────────────────────
# 7. RESULT PRINTER
# ─────────────────────────────────────────────
def print_results(result: dict):
    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  TASK  : {result['task'][:50]:<50}║")
    print(f"║  MODEL : {result['model']:<50}║")
    print("╠" + "═" * 58 + "╣")
    for step_num, output in result["results"].items():
        print(f"\n── Step {step_num} Output ───────────────────────────────")
        print(output[:1500] + ("..." if len(output) > 1500 else ""))
    print("\n" + "╚" + "═" * 58 + "╝\n")


# ─────────────────────────────────────────────
# 8. MAIN — DEMO SCENARIOS
# ─────────────────────────────────────────────
def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║    MULTI-AGENT TEAM — Groq LLaMA 3.3 70B              ║")
    print("║    Agents: Supervisor · Research · Summarize · Email   ║")
    print("╚" + "═" * 58 + "╝")

    # ── Print SMTP status at startup so you know if email is configured ──
    print("\n📧 Email Configuration Status:")
    if SMTP_USER and SMTP_PASSWORD:
        print(f"   ✅ SMTP configured — emails WILL be sent")
        print(f"   From : {SMTP_USER}")
        print(f"   Host : {SMTP_HOST}:{SMTP_PORT}")
    else:
        print("   ⚠  SMTP not configured — emails will be DRAFTED ONLY (not sent)")
        print("   To enable sending, add to your .env file:")
        print("     SMTP_HOST=smtp.gmail.com")
        print("     SMTP_PORT=587")
        print("     SMTP_USER=yourname@gmail.com")
        print("     SMTP_PASSWORD=<your 16-char Gmail App Password>")
    print()

    # ── Participants list — UPDATE THESE WITH REAL EMAILS ────────────────
    participants = [
        "recipient1@gmail.com",   # ← replace with real email addresses
        "recipient2@gmail.com",   # ← replace with real email addresses
    ]

    # ── Boot supervisor ───────────────────────────────────────────────────
    supervisor = SupervisorAgent(recipients=participants)

    # ════════════════════════════════════════════════════════════════════
    # SCENARIO 1 — Research → Summarize → Email
    # ════════════════════════════════════════════════════════════════════
    print("\n\n▶ SCENARIO 1: Research + Summarize + Email")
    result1 = supervisor.run(
        user_task=(
            "Research the topic 'Generative AI trends in 2025', "
            "summarize the key findings, and email the summary to the participants."
        ),
        recipients=participants,
    )
    print_results(result1)

    # ════════════════════════════════════════════════════════════════════
    # SCENARIO 2 — Research Only
    # ════════════════════════════════════════════════════════════════════
    print("\n\n▶ SCENARIO 2: Research Only")
    result2 = supervisor.run(
        user_task="Research the latest advancements in Multi-Agent AI Systems.",
    )
    print_results(result2)

    # ════════════════════════════════════════════════════════════════════
    # SCENARIO 3 — Summarize content → Email
    # ════════════════════════════════════════════════════════════════════
    print("\n\n▶ SCENARIO 3: Summarize Existing Content + Email")
    existing_content = """
    Azure AI Foundry is Microsoft's unified platform for building, deploying, and
    managing AI applications at enterprise scale. It integrates Azure OpenAI Service,
    Azure AI Search, Azure Machine Learning, and responsible AI tools under one platform.
    Key capabilities include: a Model Catalog with 1600+ models from OpenAI, Meta, Mistral,
    and Hugging Face; Prompt Flow for orchestrating multi-step AI pipelines; Azure Logic Apps
    integration for connecting agents to enterprise workflows; content safety filters;
    and AI-assisted evaluation. Deployment options span Standard, Batch, and Provisioned
    tiers across Global, Data Zones, and Regional locations, giving enterprises full
    control over cost, latency, and data residency compliance.
    """
    result3 = supervisor.run(
        user_task=(
            f"Summarize the following content and email the summary to participants:\n\n"
            f"{existing_content}"
        ),
        recipients=participants,
    )
    print_results(result3)

    # ════════════════════════════════════════════════════════════════════
    # SCENARIO 4 — Interactive Mode
    # ════════════════════════════════════════════════════════════════════
    print("\n\n▶ SCENARIO 4: Interactive Mode")
    print("─" * 50)
    user_input = input("Enter your task (or press Enter to skip): ").strip()

    if user_input:
        raw_emails = input(
            "Recipient emails (comma-separated, or Enter for defaults): "
        ).strip()
        emails = (
            [e.strip() for e in raw_emails.split(",") if e.strip()]
            or participants
        )
        result4 = supervisor.run(user_task=user_input, recipients=emails)
        print_results(result4)
    else:
        print("Skipping interactive mode.\n")

    print("\n✅ All scenarios complete.")
    print(f"   Powered by Groq · Model: {GROQ_MODEL}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
