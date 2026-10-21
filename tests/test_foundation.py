import sqlite3

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.database.connection import Database
from src.knowledge.vector_store import KnowledgeStore
from src.models import AgentInput
from src.services.agents import AgentDeletionError, AgentService


class Cleanup:
    fail = False

    def delete_for_agent(self, agent_id):
        if self.fail:
            raise RuntimeError("unavailable")


@pytest.fixture
def service(tmp_path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    return AgentService(database, Cleanup())


def payload(name="HR"):
    return AgentInput(
        name=name,
        description="Kebijakan perusahaan",
        instructions="Gunakan dokumen.",
        provider="litellm",
        model="ollama_chat/llama3.1:8b",
    )


def test_agent_changes_persist_across_database_instances(service):
    agent = service.create(payload())
    service.update(agent.id, payload("Legal"))
    reopened = AgentService(Database(service.database.path), Cleanup())
    assert reopened.get(agent.id).name == "Legal"
    assert len(reopened.list_agents()) == 1


def test_delete_cascades_and_rejects_cross_agent_conversation(service):
    agent = service.create(payload())
    other = service.create(payload("Legal"))
    with service.database.connect() as connection:
        connection.execute(
            "INSERT INTO documents(id,agent_id,filename,file_type,status) VALUES(?,?,?,?,?)",
            ("doc", agent.id, "policy.pdf", "pdf", "ready"),
        )
        connection.execute("INSERT INTO conversations(id,agent_id) VALUES(?,?)", ("chat", agent.id))
        connection.execute(
            "INSERT INTO messages(conversation_id,agent_id,role,content) VALUES(?,?,?,?)",
            ("chat", agent.id, "user", "Pertanyaan"),
        )
    with pytest.raises(sqlite3.IntegrityError), service.database.connect() as connection:
        connection.execute(
            "INSERT INTO messages(conversation_id,agent_id,role,content) VALUES(?,?,?,?)",
            ("chat", other.id, "user", "Lintas agent"),
        )
    service.delete(agent.id)
    with service.database.connect() as connection:
        for table in ("documents", "conversations", "messages"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert service.get(other.id).name == "Legal"


def test_failed_vector_cleanup_deactivates_agent_and_can_retry(service):
    agent = service.create(payload())
    service.knowledge.fail = True
    with pytest.raises(AgentDeletionError):
        service.delete(agent.id)
    assert service.list_agents() == []
    assert service.get(agent.id, include_deleting=True).status == "deleting"
    service.knowledge.fail = False
    service.delete(agent.id)
    assert service.list_agents(include_deleting=True) == []


def test_vector_cleanup_preserves_other_agent_and_validates_embedding(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    store = KnowledgeStore(settings)
    vector = [1.0] + [0.0] * (settings.embedding_dimension - 1)
    store.collection.add(
        ids=["hr", "legal"],
        embeddings=[vector, vector],
        metadatas=[{"agent_id": "hr"}, {"agent_id": "legal"}],
    )
    store.delete_for_agent("hr")
    reopened = KnowledgeStore(settings)
    assert reopened.collection.get()["ids"] == ["legal"]
    with pytest.raises(ValueError, match="embedding"):
        KnowledgeStore(settings.model_copy(update={"embedding_model": "another-model"}))


def test_configuration_rejects_invalid_chunks_and_hides_keys(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, openai_api_key="test-placeholder")
    assert "test-placeholder" not in repr(settings)
    assert Settings(_env_file=None, openai_api_key="").openai_api_key is None
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size_tokens=100, chunk_overlap_tokens=100)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sqlite_filename="../outside.db")
