import os
import re
import logging
from pydantic import BaseModel, Field
from google.adk.agents.llm_agent import LlmAgent

# Configure JSON-style structured logging (Pillar 4)
logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger("healthcare_assistant")

# --- Pillar 4: PII Redaction Filter ---
def redact_pii(text: str) -> str:
    """
    Scrubs sensitive private details (SSNs, Phone Numbers, Emails) to protect patient privacy.
    """
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    phone_pattern = r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    redacted = re.sub(ssn_pattern, "[REDACTED SSN]", text)
    redacted = re.sub(phone_pattern, "[REDACTED PHONE]", redacted)
    redacted = re.sub(email_pattern, "[REDACTED EMAIL]", redacted)
    return redacted

# --- Pillar 1: Explicit JSON Input Schemas ---
class ClinicalNotesInput(BaseModel):
    patient_id: str = Field(..., description="The unique clinical identifier of the patient, e.g., 'pt-9821'.")
    visit_date: str = Field(..., description="The date of the clinical visit in YYYY-MM-DD format, e.g., '2026-11-15'.")

class AppointmentInput(BaseModel):
    patient_id: str = Field(..., description="The unique identifier of the patient, e.g., 'pt-9821'.")

# --- Pillar 1 & 3: Secure Tools with Guided Error Handling & HITL Checkpoints ---
def read_clinical_notes(patient_id: str, visit_date: str) -> str:
    """
    Securely reads and extracts doctor's notes, lab results, and active diagnoses 
    from the patient's local clinical record file.

    Args:
        patient_id (str): The unique clinical identifier of the patient, e.g., 'pt-9821'.
        visit_date (str): The date of the clinical visit in YYYY-MM-DD format, e.g., '2026-11-15'.
    """
    logger.info(f"INTENT: Attempting to retrieve clinical record for Patient: {patient_id} on Date: {visit_date}")
    filename = f"clinical_notes_{patient_id}_{visit_date}.txt"
    try:
        with open(filename, "r") as file:
            content = file.read()
            logger.info(f"OUTCOME: Successfully retrieved clinical record for Patient: {patient_id}")
            return f"--- CLINICAL RECORD START ---\n{content}\n--- CLINICAL RECORD END ---"
    except FileNotFoundError:
        err_msg = f"Error: The clinical record for Patient ID '{patient_id}' on date '{visit_date}' could not be found. Please verify the date."
        logger.warning(f"OUTCOME: Failed retrieval. {err_msg}")
        return err_msg

def get_upcoming_appointments(patient_id: str) -> str:
    """
    Retrieves the date, time, doctor, and department for the patient's upcoming appointments.

    Args:
        patient_id (str): The unique identifier of the patient, e.g., 'pt-9821'.
    """
    logger.info(f"INTENT: Querying upcoming appointments for Patient: {patient_id}")
    filename = f"appointments_{patient_id}.json"
    try:
        with open(filename, "r") as file:
            import json
            appointments = json.load(file)
            logger.info(f"OUTCOME: Successfully retrieved appointments for Patient: {patient_id}")
            return json.dumps(appointments, indent=2)
    except FileNotFoundError:
        err_msg = f"Error: No upcoming appointment records found for Patient ID '{patient_id}'."
        logger.warning(f"OUTCOME: Failed retrieval. {err_msg}")
        return err_msg

def draft_and_approve_family_update(patient_id: str, summary_of_visit: str, auto_approve: bool = False) -> dict:
    """
    Drafts a friendly health update for family members and halts execution for explicit Human-in-the-Loop approval.

    Args:
        patient_id (str): The unique identifier of the patient, e.g., 'pt-9821'.
        summary_of_visit (str): A plain-English, layman's summary of the clinical visit.
        auto_approve (bool): Set to True to skip manual terminal prompt for testing.
    """
    logger.info(f"INTENT: Drafting family update message for Patient: {patient_id}")
    family_message_draft = f"Subject: Quick Health Update 🩺❤️\n\nHi everyone,\n\nI wanted to share an update: {summary_of_visit}\n\nLove,\nAlex"
    
    print("\n" + "="*50)
    print("📢 HUMAN-IN-THE-LOOP CHECKPOINT")
    print(family_message_draft)
    print("="*50)

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
    else:
        logger.warning("OUTCOME: Family update rejected by human user.")
        return {"status": "REJECTED", "final_draft": None}

# --- Pillar 2: System Prompt ---
healthcare_instructions = """
You are a warm, supportive Personal Healthcare Assistant. Your core rules are:
1. Translate complex clinical records into simple, compassionate layman's terms.
2. Help the patient prepare for upcoming visits with 2-3 personalized questions.
3. NEVER prescribe, diagnose, or make up medical advice. Instantly instruct the patient to contact their doctor.
4. Protect user privacy.
"""

# --- Pillar 3: Strategic Model Routing (Sub-Agents) ---
# Sub-Agent A: Lightweight data gathering (Flash-lite for low latency)
records_agent = LlmAgent(
    model='gemini-3.5-flash-lite',
    name='records_retriever',
    instruction="Extract raw patient histories, dates, and upcoming schedules precisely.",
    tools=[read_clinical_notes, get_upcoming_appointments]
)

# Sub-Agent B: Heavy clinical reasoning and translation (Pro for medical comprehension)
translator_agent = LlmAgent(
    model='gemini-1.5-pro',
    name='clinical_translator',
    instruction="Translate medical jargon into compassionate, simple terms and draft custom doctor-visit prep guides.",
    tools=[]
)

# Coordinator Root Agent
root_agent = LlmAgent(
    model='gemini-3.5-flash-lite',
    name='healthcare_coordinator',
    instruction=healthcare_instructions,
    tools=[draft_and_approve_family_update],
    sub_agents=[records_agent, translator_agent]
)
