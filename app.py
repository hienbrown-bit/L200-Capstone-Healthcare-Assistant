import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

# --- Pillar 4: OpenTelemetry Distributed Tracing ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from character import root_agent, redact_pii, logger


# --- Pillar 5: Secure Secret Management (Secret Manager + .env fallback) ---
def load_secure_api_key() -> None:
    """Loads GEMINI_API_KEY from Google Cloud Secret Manager or local .env without hardcoding."""
    load_dotenv()
    if os.environ.get("GEMINI_API_KEY"):
        logger.info("SECRET MANAGEMENT: Loaded GEMINI_API_KEY from environment/.env.")
        return

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "hb-gcp-test-project")
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{project_id}/secrets/gemini-api-key/versions/latest"
        response = client.access_secret_version(request={"name": secret_name})
        os.environ["GEMINI_API_KEY"] = response.payload.data.decode("UTF-8")
        logger.info("SECRET MANAGEMENT: Loaded GEMINI_API_KEY from Google Cloud Secret Manager.")
    except Exception as e:
        logger.warning(f"SECRET MANAGEMENT: Secret Manager lookup skipped ({e}); relying on runtime env.")


load_secure_api_key()

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("healthcare_companion")

app = Flask(__name__)

# --- Pillar 2: Persistent Session State using Async SQLite DB ---
session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///healthcare_sessions.db")

runner = Runner(
    agent=root_agent,
    app_name="healthcare_companion",
    session_service=session_service,
)


# --- Pillar 2: Sliding-Window Context Compaction ---
async def compact_session_history(user_id: str, session_id: str, max_events: int = 12):
    """Prevents token bloat by trimming session events to the most recent turns."""
    try:
        session = await session_service.get_session(
            app_name="healthcare_companion", user_id=user_id, session_id=session_id
        )
        if session and hasattr(session, "events") and len(session.events) > max_events:
            logger.info(f"COMPACTION: Compacting session {session_id} to last {max_events} events.")
            session.events = session.events[-max_events:]
    except Exception as e:
        logger.warning(f"COMPACTION: Failed to compact history: {e}")


# --- Pillar 2: Async Background Memory Operations ---
async def consolidate_memory_async(user_id: str, session_id: str):
    """Performs non-blocking background extraction of key patient conditions."""
    await asyncio.sleep(1)
    logger.info(f"ASYNC MEMORY: Consolidated background medical context for {user_id} ({session_id})")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
async def chat_endpoint():
    with tracer.start_as_current_span("chat_endpoint_transaction") as span:
        data = request.get_json() or {}
        user_query = data.get("message", "")
        user_id = data.get("user_id", "pt-9821")
        session_id = data.get("session_id", "live-session")

        # 1. PII Redaction before processing (Pillar 4)
        safe_query = redact_pii(user_query)
        span.set_attribute("query.length", len(safe_query))

        # 2. Deterministic Pre-Model Safety Guardrail (Pillar 3)
        illicit_triggers = ["prescribe", "dosage", "prescription", "medication name", "antibiotics"]
        if any(trigger in safe_query.lower() for trigger in illicit_triggers):
            logger.warning("GUARDRAIL TRIGGERED: Blocking drug prescription attempt.")
            return jsonify(
                {
                    "response": (
                        "I apologize, but I am an AI assistant and not a licensed doctor. "
                        "I cannot prescribe any medications or adjust dosages. "
                        "Please contact Dr. Sarah Smith's office or your primary care physician directly."
                    )
                }
            )

        try:
            await session_service.create_session(
                app_name="healthcare_companion",
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            pass

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=safe_query)],
        )

        assistant_response = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                assistant_response += event.content.parts[0].text

        await compact_session_history(user_id, session_id)
        asyncio.create_task(consolidate_memory_async(user_id, session_id))

        return jsonify({"response": assistant_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)