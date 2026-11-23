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
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("healthcare_assistant")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False


# --- Pillar 4: PII Redaction Filter ---
def redact_pii(text: str) -> str:
    """Scrubs sensitive private details (SSNs, Phone Numbers, Emails) to protect patient privacy."""
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    phone_pattern = r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

    redacted = re.sub(ssn_pattern, "[REDACTED SSN]", text)
    redacted = re.sub(phone_pattern, "[REDACTED PHONE]", redacted)
    redacted = re.sub(email_pattern, "[REDACTED EMAIL]", redacted)
    return redacted


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
    """
    Securely reads and extracts doctor's notes, lab results, and active diagnoses
    from the patient's local clinical record file.

    Args:
        patient_id (str): The unique clinical identifier of the patient, e.g., 'pt-9821'.
        visit_date (str): The date of the clinical visit in YYYY-MM-DD format, e.g., '2026-11-15'.

    Returns:
        str: Clinical record text or guided error recovery guidance.
    """
    try:
        validated = ClinicalNotesInput(patient_id=patient_id, visit_date=visit_date)
    except ValidationError as ve:
        return f"Validation Error: Invalid input format ({ve}). Please provide patient_id like 'pt-9821' and visit_date as 'YYYY-MM-DD'."

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
    """
    Retrieves the date, time, doctor, and department for the patient's upcoming appointments.

    Args:
        patient_id (str): The unique identifier of the patient, e.g., 'pt-9821'.

    Returns:
        str: JSON string of upcoming appointments or guided error recovery message.
    """