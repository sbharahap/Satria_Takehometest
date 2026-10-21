import logging
from typing import Protocol
from uuid import uuid4

from src.database.connection import Database
from src.models import Agent, AgentInput

logger = logging.getLogger(__name__)


class KnowledgeCleanup(Protocol):
    def delete_for_agent(self, agent_id: str) -> None: ...


class AgentNotFoundError(ValueError):
    pass


class AgentDeletionError(RuntimeError):
    pass


class AgentService:
    def __init__(self, database: Database, knowledge: KnowledgeCleanup):
        self.database = database
        self.knowledge = knowledge

    def list_agents(self, *, include_deleting: bool = False) -> list[Agent]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agents "
                + ("" if include_deleting else "WHERE status = 'active' ")
                + "ORDER BY created_at, id"
            ).fetchall()
        return [Agent(**dict(row)) for row in rows]

    def get(self, agent_id: str, *, include_deleting: bool = False) -> Agent:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if row is None or (row["status"] != "active" and not include_deleting):
            raise AgentNotFoundError("Agent tidak ditemukan atau sedang dihapus.")
        return Agent(**dict(row))

    def create(self, agent: AgentInput) -> Agent:
        agent_id = uuid4().hex
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO agents (id,name,description,instructions,provider,model) "
                "VALUES (?,?,?,?,?,?)",
                (
                    agent_id,
                    agent.name,
                    agent.description,
                    agent.instructions,
                    agent.provider,
                    agent.model,
                ),
            )
        logger.info("agent_created agent_id=%s provider=%s", agent_id, agent.provider)
        return self.get(agent_id)

    def update(self, agent_id: str, agent: AgentInput) -> Agent:
        with self.database.connect() as connection:
            result = connection.execute(
                "UPDATE agents SET name=?,description=?,instructions=?,provider=?,model=?,"
                "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=? AND status='active'",
                (
                    agent.name,
                    agent.description,
                    agent.instructions,
                    agent.provider,
                    agent.model,
                    agent_id,
                ),
            )
            if result.rowcount == 0:
                raise AgentNotFoundError("Agent tidak ditemukan atau sedang dihapus.")
        return self.get(agent_id)

    def delete(self, agent_id: str) -> None:
        # Mark inactive before cross-store cleanup; retries are idempotent.
        with self.database.connect() as connection:
            result = connection.execute(
                "UPDATE agents SET status='deleting' WHERE id=?", (agent_id,)
            )
            if result.rowcount == 0:
                raise AgentNotFoundError("Agent tidak ditemukan.")
        try:
            self.knowledge.delete_for_agent(agent_id)
        except Exception:
            logger.error("agent_cleanup_failed agent_id=%s", agent_id)
            raise AgentDeletionError(
                "Pembersihan knowledge gagal. Agent dinonaktifkan; coba hapus kembali."
            ) from None
        with self.database.connect() as connection:
            connection.execute("DELETE FROM agents WHERE id=?", (agent_id,))
        logger.info("agent_deleted agent_id=%s", agent_id)
