import os
import re
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.genai import types


# --- Pillar 4: True Structured JSON Logging (using json.dumps) ---
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("healthcare_assistant")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False


# --- Pillar 4: PII Redaction Filter ---
def redact_pii(text: str) -> str:
    """Scrubs sensitive private details (SSNs, Phone Numbers, Emails) to protect patient privacy."""
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED SSN]", text)
    text = re.sub(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", "[REDACTED PHONE]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED EMAIL]", text)
    return text


# --- Pillar 1: Explicit JSON Input Schemas ---
class ClinicalNotesInput(BaseModel):
    patient_id: str = Field(..., pattern=r"^pt-\d+$", description="Patient ID, e.g., 'pt-9821'.")
    visit_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Visit date in YYYY-MM-DD format.")


class AppointmentInput(BaseModel):
    patient_id: str = Field(..., pattern=r"^pt-\d+$", description="Patient ID, e.g., 'pt-9821'.")


class FamilyDraftInput(BaseModel):
    patient_id: str = Field(..., pattern=r"^pt-\d+$", description="Patient ID, e.g., 'pt-9821'.")
    summary_of_visit: str = Field(..., min_length=5, description="Plain-English clinical summary.")


# --- Pillar 1 & 3: Secure Tools Actively Enforcing Pydantic Validation ---
def read_clinical_notes(patient_id: str, visit_date: str) -> str:
    """Securely reads and extracts doctor's notes, lab results, and active diagnoses.

    Args:
        patient_id (str): The unique clinical identifier of the patient, e.g., 'pt-9821'.
        visit_date (str): The date of the clinical visit in YYYY-MM-DD format, e.g., '2026-11-15'.

    Returns:
        str: Clinical record text or guided error recovery guidance.
    """
    try:
        validated = ClinicalNotesInput(patient_id=patient_id, visit_date=visit_date)
    except ValidationError as ve:
        return f"Validation Error: Invalid input format ({ve}). Use patient_id like 'pt-9821' and visit_date as 'YYYY-MM-DD'."

    logger.info(f"INTENT: Retrieving clinical record for {validated.patient_id} on {validated.visit_date}")
    filename = f"clinical_notes_{validated.patient_id}_{validated.visit_date}.txt"
    try:
        with open(filename, "r") as file:
            content = redact_pii(file.read())
            logger.info(f"OUTCOME: Retrieved clinical record for {validated.patient_id}")
            return f"--- CLINICAL RECORD START ---\n{content}\n--- CLINICAL RECORD END ---"
    except FileNotFoundError:
        err_msg = f"Error: Clinical record for '{validated.patient_id}' on '{validated.visit_date}' not found. Verify date and patient ID."
        logger.warning(f"OUTCOME: {err_msg}")
        return err_msg


def get_upcoming_appointments(patient_id: str) -> str:
    """Retrieves the date, time, doctor, and department for upcoming appointments.

    Args:
        patient_id (str): The unique identifier of the patient, e.g., 'pt-9821'.

    Returns:
        str: JSON string of upcoming appointments or guided error recovery message.
    """
    try:
        validated = AppointmentInput(patient_id=patient_id)
    except ValidationError as ve:
        return f"Validation Error: Invalid patient_id ({ve}). Expected format like 'pt-9821'."

    logger.info(f"INTENT: Querying upcoming appointments for {validated.patient_id}")
    filename = f"appointments_{validated.patient_id}.json"
    try:
        with open(filename, "r") as file:
            appointments = json.load(file)
            logger.info(f"OUTCOME: Retrieved appointments for {validated.patient_id}")
            return json.dumps(appointments, indent=2)
    except FileNotFoundError:
        err_msg = f"Error: No upcoming appointment records found for '{validated.patient_id}'."
        logger.warning(f"OUTCOME: {err_msg}")
        return err_msg


def draft_and_approve_family_update(patient_id: str, summary_of_visit: str, auto_approve: bool = False) -> dict:
    """Drafts a friendly health update for family members and halts for explicit Human-in-the-Loop approval.

    Args:
        patient_id (str): The unique identifier of the patient, e.g., 'pt-9821'.
        summary_of_visit (str): Plain-English summary of the clinical visit.
        auto_approve (bool): Set to True to skip manual terminal prompt during automated runs.

    Returns:
        dict: Approval status and final drafted text.
    """
    try:
        validated = FamilyDraftInput(patient_id=patient_id, summary_of_visit=summary_of_visit)
    except ValidationError as ve:
        return {"status": "ERROR", "message": f"Invalid draft inputs: {ve}", "final_draft": None}

    logger.info(f"INTENT: Drafting family update for {validated.patient_id}")
    safe_summary = redact_pii(validated.summary_of_visit)
    family_message_draft = (
        f"Subject: Quick Health Update\n\nHi everyone,\n\n"
        f"I wanted to share an update: {safe_summary}\n\nLove,\nAlex"
    )
    print("\n" + "=" * 50)
    print("HUMAN-IN-THE-LOOP CHECKPOINT")
    print(family_message_draft)
    print("=" * 50)

    if auto_approve:
        decision = "YES"
    else:
        try:
            decision = input("Do you approve sending/saving this draft? (YES/NO): ").strip().upper()
        except Exception:
            decision = "NO"

    if decision == "YES":
        logger.info("OUTCOME: Family update approved by human user.")
        return {"status": "APPROVED", "final_draft": family_message_draft}
    logger.warning("OUTCOME: Family update rejected by human user.")
    return {"status": "REJECTED", "final_draft": None}


# --- Pillar 2: System Prompt ---
healthcare_instructions = """You are a warm, supportive Personal Healthcare Assistant. Your core rules are:
1. Translate complex clinical records into simple, compassionate layman's terms.
2. Help the patient prepare for upcoming visits with 2-3 personalized questions.
3. NEVER prescribe, diagnose, or make up medical advice. Instantly instruct the patient to contact their doctor.
4. Protect user privacy."""

# --- Pillar 3: Native Gemini Safety Settings & ADK Policy Callback Plugin ---
native_safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]


def medical_policy_plugin(callback_context, llm_request):
    """ADK before_model_callback policy plugin enforcing clinical scope & safety screening."""
    last_text = ""
    if llm_request.contents:
        for part in llm_request.contents[-1].parts or []:
            if getattr(part, "text", None):
                last_text += part.text.lower()

    restricted_intents = ["prescribe", "dosage", "prescription", "antibiotics", "diagnose me"]
    if any(term in last_text for term in restricted_intents):
        logger.warning("ADK POLICY PLUGIN: Blocked unauthorized medical prescription/diagnosis request.")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text="I apologize, but I am an AI healthcare assistant and not a licensed physician. "
                         "I cannot prescribe medications, adjust dosages, or diagnose conditions. "
                         "Please consult Dr. Sarah Smith or your primary care provider directly."
                )],
            )
        )
    return None


# --- Pillar 3: Strategic Model Routing (Sub-Agents) ---
records_agent = LlmAgent(
    model="gemini-3.5-flash-lite",
    name="records_retriever",
    instruction="Extract raw patient histories, dates, and upcoming schedules precisely.",
    tools=[read_clinical_notes, get_upcoming_appointments],
)

translator_agent = LlmAgent(
    model="gemini-1.5-pro",
    name="clinical_translator",
    instruction="Translate medical jargon into compassionate, simple terms and draft custom doctor-visit prep guides.",
    tools=[],
)

root_agent = LlmAgent(
    model="gemini-3.5-flash-lite",
    name="healthcare_coordinator",
    instruction=healthcare_instructions,
    tools=[draft_and_approve_family_update],
    sub_agents=[records_agent, translator_agent],
    generate_content_config=types.GenerateContentConfig(safety_settings=native_safety_settings),
    before_model_callback=medical_policy_plugin,
)
