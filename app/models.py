from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Event structure from offense
class TopEvent(BaseModel):
    qid: Optional[int] = None
    low_level_category: Optional[str] = None
    high_level_category: Optional[str] = None
    sourceip: Optional[str] = None
    sourceport: Optional[int] = None
    destinationip: Optional[str] = None
    destinationport: Optional[int] = None
    username: Optional[str] = None
    event_outcome: Optional[str] = None
    PROTOCOLNAME: Optional[str] = None
    count: Optional[int] = None

    model_config = ConfigDict(extra="allow")

# offense structure
class Offense(BaseModel):
    id: int | str = Field(..., description="Unique offense identifier")
    description: Optional[str] = Field(None, description="Triggered rule name / offense description")
    magnitude: Optional[int] = None
    severity: Optional[int] = None
    credibility: Optional[int] = None
    relevance: Optional[int] = None

    offenseSource: Optional[str] = Field(None, description="Raw attacker identifier, usually an IP")
    attacker: Optional[str] = None
    attackerDescription: Optional[str] = Field(None, description="Attacker IP plus resolved hostname/context")

    target: Optional[str] = None
    targetDescription: Optional[str] = Field(None, description="Target IP plus resolved hostname/context")
    targetNetwork: Optional[str] = None
    domainName: Optional[str] = None

    eventCount: Optional[int] = None
    categoryCount: Optional[int] = None
    eventDescription: Optional[str] = None

    startTime: Optional[str] = None
    endTime: Optional[str] = None
    formattedDuration: Optional[str] = None

    top_events: list[TopEvent] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    # Text version of the offence quiery
    def to_search_query(self) -> str:
        parts: list[str] = []

        if self.description:
            parts.append(self.description.replace("_", " "))
        if self.attackerDescription or self.attacker:
            parts.append(f"Attacker: {self.attackerDescription or self.attacker}")
        if self.targetDescription or self.target:
            parts.append(f"Target: {self.targetDescription or self.target}")
        if self.targetNetwork:
            parts.append(f"Target network/zone: {self.targetNetwork}")

        if self.top_events:
            categories = sorted({e.low_level_category for e in self.top_events if e.low_level_category})
            if categories:
                parts.append("Event categories: " + ", ".join(categories))

            ports = sorted({str(e.destinationport) for e in self.top_events if e.destinationport is not None})
            if ports:
                parts.append("Destination ports: " + ", ".join(ports))

            outcomes = sorted({e.event_outcome for e in self.top_events if e.event_outcome})
            if outcomes:
                parts.append("Outcomes observed: " + ", ".join(outcomes))

            usernames = sorted({e.username for e in self.top_events if e.username})
            if usernames:
                parts.append("Target usernames: " + ", ".join(usernames))

        return ". ".join(p for p in parts if p)

class Classification(str, Enum):
    TRUE_POSITIVE = "TP"
    FALSE_POSITIVE = "FP"

class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

# A knowledge-base chunk that was retrieved
class KnowledgeBaseRef(BaseModel):
    source: str
    chunk_id: str
    score: float

# Structured output
class AISummary(BaseModel):
    offense_id: str
    summary: str = Field(..., description="Summary of what happened")
    classification: Classification
    classification_reason: str = Field(..., description="Short justification for the FP/TP call")
    recommended_action: str = Field(..., description="Recommended next step or playbook reference")
    playbook_reference: Optional[str] = Field(None, description="Matched playbook file/title, if any")
    confidence: ConfidenceLevel
    supporting_context: list[KnowledgeBaseRef] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: Optional[str] = None
