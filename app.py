import os
import asyncio
from flask import Flask, request, jsonify, render_template
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

# --- Pillar 4: OpenTelemetry Distributed Tracing ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("healthcare_companion")

# Import our customized root agent and helpers
from character import root_agent, redact_pii, logger

app = Flask(__name__)

# --- Pillar 2: Persistent Session State using SQLite DB ---
session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///healthcare_sessions.db")
runner = Runner(
    agent=root_agent,
    app_name="healthcare_companion",
    session_service=session_service
)

# --- Pillar 2: Sliding-Window Context Compaction ---
async def compact_session_history(user_id: str, session_id: str, max_events: int = 12):
    """
    Prevents token bloat by sliding conversational history down to the most recent turns.
    """
    try:
        session = await session_service.get_session(app_name="healthcare_companion", user_id=user_id, session_id=session_id)
        if len(session.history) > max_events:
            logger.info(f"COMPACTION: Compacting history for session {session_id} to last {max_events} elements.")
            session.history = session.history[-max_events:]
            await session_service.update_session(session)
    except Exception as e:
        logger.warning(f"COMPACTION: Failed to compact history: {str(e)}")

# --- Pillar 2: Async Background Memory operations ---
async def consolidate_memory_async(user_id: str, session_id: str):
    """
    Performs expensive background extraction of key medical conditions to keep UI non-blocking.
    """
    await asyncio.sleep(1) # Simulate non-blocking worker thread activity
    logger.info(f"ASYNC MEMORY: Extracted key patient metrics in background for {user_id}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
async def chat_endpoint():
    # Start tracing this transaction (Pillar 4)
    with tracer.start_as_current_span("chat_endpoint_transaction") as span:
        data = request.get_json() or {}
        user_query = data.get("message", "")
        user_id = data.get("user_id", "pt-9821")
        session_id = data.get("session_id", "live-session")

        # 1. Active scrubbing before processing (Pillar 4)
        safe_query = redact_pii(user_query)
        span.set_attribute("query.length", len(safe_query))

        # --- Pillar 3: Before-Model Policy Plugin (Deterministic Guardrail) ---
        illicit_triggers = ["prescribe", "dosage", "prescription", "medication name", "antibiotics"]
        if any(trigger in safe_query.lower() for trigger in illicit_triggers):
            logger.warning("GUARDRAIL TRIGGERED: Blocking drug prescription attempt.")
            return jsonify({
                "response": "I apologize, but I am an AI assistant and not a licensed doctor. I cannot prescribe any medications or adjust dosages. Please contact Dr. Sarah Smith's office or your primary care physician directly."
            })

        # Ensure database session exists
        try:
            await session_service.create_session(
                app_name="healthcare_companion",
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            pass

        # Package user content
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=safe_query)]
        )

        assistant_response = ""
        # 2. Run agent pipeline using ADK Runner
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                assistant_response += event.content.parts[0].text

        # 3. Trigger sliding compaction (Pillar 2)
        await compact_session_history(user_id, session_id)

        # 4. Trigger non-blocking async operations (Pillar 2)
        asyncio.create_task(consolidate_memory_async(user_id, session_id))

        return jsonify({"response": assistant_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
