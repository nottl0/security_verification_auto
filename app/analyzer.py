import os
import json
from pathlib import Path

import google.genai as genai
from google.genai import types

from app.models import AISummary, Classification, ConfidenceLevel, KnowledgeBaseRef, Offense
from app.vector_store import RetrievedChunk, vector_store

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("TOP_K", "3"))

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_KB_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", BASE_DIR / "knowledge_base")) / "static"

BASE_SYSTEM_PROMPT = """You are an expert Security Analyst. Your role is to accurately analyze a security offense alert using the context provided from the internal knowledge base playbooks.

CRITICAL INSTRUCTIONS:
1. Base your classification and recommended actions primarily on the provided knowledge-base context and the global infrastructure boundaries provided below. If no relevant context is found, rely on industry standard best practices for the given attack vector.
2. Ensure the output strictly conforms to the JSON schema requested. Do not include markdown formatting or wrapper blocks (e.g., do not wrap in ```json ... 
```) because the application expects a clean JSON string directly.
3. Knowledge base documents whose filename, title, source, or playbook name contains
  'FP', 'FALSE_POSITIVE', or similar terminology represent documented false-positive scenarios. If these files are referenced for a decision, it should be a FP.

You must return a valid JSON object matching the following structure:
{
  "summary": "A concise, 2-3 sentence technical overview explaining what happened, the source/destination entities, and the immediate impact.",
  "classification": "Must be exactly 'TP' for true positive or 'FP' for false positive.",
  "classification_reason": "A highly detailed breakdown explaining WHY this classification was chosen, citing specific indicators or lack of evidence.",
  "recommended_action": "Clear, actionable step-by-step mitigation instructions for a junior analyst to follow.",
  "playbook_reference": "The name or ID of the playbook used from the knowledge base, or null if no matching playbook was found.",
  "confidence": "Must match one of: 'HIGH', 'MEDIUM', 'LOW'."
}"""

# Get the static context that will always be used
def get_static_context() -> str:
    static_context = ""
    if STATIC_KB_DIR.exists():
        for filename in os.listdir(STATIC_KB_DIR):
            file_path = STATIC_KB_DIR / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    static_context += f"\n\n=== GLOBAL INFRASTRUCTURE POLICY: {filename} ===\n{content}"
                except Exception as e:
                    pass
    return static_context

def build_user_prompt(offense: Offense, chunks: list[RetrievedChunk]) -> str:
    """Build a clean user prompt focusing strictly on data inputs."""
    offense_json = json.dumps(offense.model_dump(), indent=2, default=str)

    if chunks:
        context_lines = [f"[{chunk.source}]\n{chunk.text}" for chunk in chunks]
        context_block = "\n\n---\n\n".join(context_lines)
    else:
        context_block = "(no relevant knowledge-base context found)"

    return f"""
[Offense Data]:
{offense_json}

[Retrieved Knowledge Base Context]:
{context_block}
"""

def call_llm(prompt: str) -> dict:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    full_system_instruction = BASE_SYSTEM_PROMPT + get_static_context()

    config = types.GenerateContentConfig(
        system_instruction=full_system_instruction,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=config,
    )
    
    return json.loads(response.text)

def _safe_classification(value: str) -> Classification:
    try:
        return Classification(value)
    except ValueError:
        return Classification.TRUE_POSITIVE

def _safe_confidence(value: str) -> ConfidenceLevel:
    try:
        return ConfidenceLevel(value)
    except ValueError:
        return ConfidenceLevel.LOW

def analyze_offense(offense: Offense) -> AISummary:
    chunks = vector_store.search(offense.to_search_query(), top_k=TOP_K)

    prompt = build_user_prompt(offense, chunks)
    raw = call_llm(prompt)

    return AISummary(
        offense_id=str(offense.id),
        summary=raw.get("summary", ""),
        classification=_safe_classification(raw.get("classification", "")),
        classification_reason=raw.get("classification_reason", ""),
        recommended_action=raw.get("recommended_action", ""),
        playbook_reference=raw.get("playbook_reference"),
        confidence=_safe_confidence(raw.get("confidence", "")),
        supporting_context=[
            KnowledgeBaseRef(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
            )
            for chunk in chunks
        ],
        model=LLM_MODEL,
    )