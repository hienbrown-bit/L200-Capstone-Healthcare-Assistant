import os
import re
import json
import logging
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ValidationError
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models import LlmResponse
from google.genai import types

# 1. Structured JSON Logger (Pillar 4)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

logger = logging.getLogger("health_assistant")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# 2. Active PII Redaction Filter (Pillar 4)
def redact_pii(text: str) -> str:
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', "[REDACTED_SSN]", text)
    text = re.sub(r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', "[REDACTED_PHONE]", text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[REDACTED_EMAIL]", text)
    return text

# 3. Constitutional System Prompt (Pillar 2)
healthcare_instructions = """
You are Baymax, a compassionate, clear, and highly supportive Personal Healthcare Companion.
Your primary goal is to help patients be better stewards of their own health information.

Your Core Rules & Responsibilities:
1. Clinical Translation: When a patient shares clinical notes or lab results, translate complex medical terms (like 'joint effusion' or 'edema') into warm, simple, layman's terms.
2. Appointment Prep: Help patients organize their thoughts before seeing a doctor. Suggest 2-3 smart, personalized questions they should bring to their upcoming visit based on their medical history or symptoms.
3. Family Updates: Help the patient draft clear, simple summaries that they can copy/paste to share with their families.
4. ABSOLUTE MEDICAL SAFETY: You are a healthcare companion, NOT a doctor. You must NEVER diagnose conditions, prescribe medications, or make up medical advice. If asked for medical diagnoses or prescriptions, politely refuse and instruct the user to consult their primary care doctor or physician.
5. Privacy Compliance: Always remind the user to keep their highly sensitive private details safe.
6. Default Patient Context: Unless the user specifies otherwise, assume Patient ID is 'pt-9821' and latest visit date is '2026-11-15'.
"""

# 4. Validated Tools with Guided Error Recovery & Intent/Outcome Logging (Pillars 1 & 4)
class ClinicalNotesInput(BaseModel):
    patient_id: str = Field(..., pattern=r"^pt-\d+$", description="Patient ID formatted as 'pt-XXXX'.")
    visit_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Visit date in 'YYYY-MM-DD' format.")

def read_clinical_notes(patient_id: str, visit_date: str) -> str:
    """Reads and extracts doctor's notes, lab results, and diagnoses from the patient's clinical record."""
    logger.info("Tool intent: read_clinical_notes", extra={"extra_data": {
        "phase": "intent", "tool": "read_clinical_notes", "patient_id": patient_id, "visit_date": visit_date
    }})
    try:
        validated = ClinicalNotesInput(patient_id=patient_id, visit_date=visit_date)
    except ValidationError as ve:
        err_msg = f"Validation Error: {ve.errors()[0]['msg']}. Ensure patient_id is 'pt-XXXX' and visit_date is 'YYYY-MM-DD'."
        logger.warning("Tool outcome: validation_failed", extra={"extra_data": {"phase": "outcome", "error": err_msg}})
        return err_msg

    filename = f"clinical_notes_{validated.patient_id}_{validated.visit_date}.txt"
    try:
        with open(filename, "r") as file:
            content = redact_pii(file.read())
            logger.info("Tool outcome: success", extra={"extra_data": {"phase": "outcome", "file": filename}})
            return f"--- CLINICAL RECORD START ---\n{content}\n--- CLINICAL RECORD END ---"
    except FileNotFoundError:
        recovery = f"Recovery Instruction: Clinical record for '{validated.patient_id}' on '{validated.visit_date}' not found. Ask the user to verify their Patient ID and visit date."
        logger.warning("Tool outcome: file_not_found", extra={"extra_data": {"phase": "outcome", "file": filename}})
        return recovery

class AppointmentInput(BaseModel):
    patient_id: str = Field(..., pattern=r"^pt-\d+$", description="Patient ID formatted as 'pt-XXXX'.")

def get_upcoming_appointments(patient_id: str) -> str:
    """Retrieves the date, time, doctor, and department for the patient's upcoming appointments."""
    logger.info("Tool intent: get_upcoming_appointments", extra={"extra_data": {
        "phase": "intent", "tool": "get_upcoming_appointments", "patient_id": patient_id
    }})
    try:
        validated = AppointmentInput(patient_id=patient_id)
    except ValidationError as ve:
        err_msg = f"Validation Error: {ve.errors()[0]['msg']}. Ensure patient_id is 'pt-XXXX'."
        logger.warning("Tool outcome: validation_failed", extra={"extra_data": {"phase": "outcome", "error": err_msg}})
        return err_msg

    filename = f"appointments_{validated.patient_id}.json"
    try:
        with open(filename, "r") as file:
            appointments = json.load(file)
            logger.info("Tool outcome: success", extra={"extra_data": {"phase": "outcome", "file": filename}})
            return redact_pii(json.dumps(appointments, indent=2))
    except FileNotFoundError:
        recovery = f"Recovery Instruction: No upcoming appointments found for '{validated.patient_id}'."
        logger.warning("Tool outcome: file_not_found", extra={"extra_data": {"phase": "outcome", "file": filename}})
        return recovery

def draft_and_approve_family_update(patient_id: str, summary_of_visit: str) -> dict:
    """Drafts a plain-English health update for family members and returns it for user review before sharing."""
    family_message_draft = (
        f"Subject: Quick Health Update\n\n"
        f"Hi everyone,\n\n"
        f"I wanted to share a quick update from my recent visit to Stanford Health Care:\n"
        f"{redact_pii(summary_of_visit)}\n\n"
        f"Love,\nAlex"
    )
    return {
        "status": "PENDING_USER_REVIEW",
        "message": "Please review this draft carefully before copying or sending it to your family.",
        "draft": family_message_draft
    }

# 5. Deterministic Medical Safety Guardrail Plugin (Pillar 3)
def medical_safety_guardrail(callback_context, llm_request):
    """Intercepts prescription or dosage requests before they reach the LLM."""
    if llm_request.contents:
        last_text = llm_request.contents[-1].parts[0].text.lower()
        forbidden_triggers = ["prescribe me", "write a prescription", "dosage should i take"]
        if any(trigger in last_text for trigger in forbidden_triggers):
            logger.warning("Guardrail triggered: prescription_request_blocked", extra={"extra_data": {
                "guardrail": "medical_safety", "action": "blocked"
            }})
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="I am your AI healthcare companion, not a licensed physician, so I cannot prescribe medications or recommend dosages. Please contact your primary care doctor or physician directly."
                    )]
                )
            )
    return None

# 6. Multi-Agent Coordinator Architecture (Pillar 3)
records_retriever_agent = LlmAgent(
    model="gemini-3.5-flash-lite",
    name="records_retriever_agent",
    description="Fast retrieval specialist that fetches raw clinical notes and upcoming appointments.",
    instruction=(
        "You are a fast medical records retrieval agent. Use `read_clinical_notes` and "
        "`get_upcoming_appointments` (defaulting to patient_id='pt-9821' and visit_date='2026-11-15' if unspecified) "
        "to fetch the exact facts requested and hand off to `clinical_translator_agent`."
    ),
    tools=[read_clinical_notes, get_upcoming_appointments]
)

clinical_translator_agent = LlmAgent(
    model="gemini-3.5-flash-lite",
    name="clinical_translator_agent",
    description="Clinical communication specialist that translates medical jargon into plain English, prepares doctor visit questions, and drafts family updates.",
    instruction=healthcare_instructions,
    tools=[draft_and_approve_family_update],
    before_model_callback=medical_safety_guardrail
)

root_agent = LlmAgent(
    model="gemini-3.5-flash-lite",
    name="healthcare_coordinator",
    instruction=(
        "You are Baymax, the Healthcare Assistant Coordinator. Delegate clinical note and appointment lookups "
        "to `records_retriever_agent`, and delegate medical jargon translation, doctor visit question prep, "
        "and family update drafts to `clinical_translator_agent`. Enforce all medical safety boundaries."
    ),
    sub_agents=[records_retriever_agent, clinical_translator_agent],
    before_model_callback=medical_safety_guardrail
)
