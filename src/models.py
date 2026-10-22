from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProviderName(StrEnum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    LITELLM = "litellm"


class AgentInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    instructions: str = Field(min_length=1, max_length=8000)
    provider: ProviderName
    model: str = Field(min_length=1, max_length=200)

    @field_validator("name", "description", "instructions", "model", mode="before")
    @classmethod
    def trim_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class Agent(AgentInput):
    model_config = ConfigDict(frozen=True)
    id: str
    status: str
    created_at: str
    updated_at: str


class Document(BaseModel):
    id: str
    agent_id: str
    filename: str
    file_type: str
    status: str
    revision: int = 1
    chunks_count: int
    error_message: str | None
    created_at: str


class Source(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float
    locator: str | None = None
    revision: int = 1

    @property
    def location(self) -> str:
        return self.locator or (f"halaman PDF {self.page}" if self.page else "Bagian dokumen")


class Citation(Source):
    tag: str
