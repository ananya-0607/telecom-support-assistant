from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class ResolutionStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    instruction: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)

class Resolution(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['suggested_resolution','insufficient_evidence']
    summary: str = Field(min_length=1)
    steps: list[ResolutionStep]
    missing_information: list[str]
    escalation: str
