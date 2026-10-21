import logging
from dataclasses import dataclass

from src.config import Settings
from src.database.connection import Database
from src.knowledge.embeddings import LocalEmbeddings
from src.knowledge.vector_store import KnowledgeStore
from src.providers.registry import ProviderRegistry
from src.services.agents import AgentService
from src.services.chat import ChatService
from src.services.documents import DocumentService


@dataclass
class Services:
    database: Database
    knowledge: KnowledgeStore
    agents: AgentService
    documents: DocumentService
    chat: ChatService


def initialize(settings: Settings) -> Services:
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    settings.ensure_directories()
    database = Database(settings.sqlite_path)
    database.initialize()
    knowledge = KnowledgeStore(settings)
    # Synchronous ingestion cannot resume work interrupted by a process restart.
    with database.connect() as connection:
        connection.execute(
            "UPDATE documents SET status='failed',error_message="
            "'Pemrosesan terhenti saat aplikasi restart. Hapus lalu unggah ulang.' "
            "WHERE status='processing'"
        )
    agents = AgentService(database, knowledge)
    embeddings = LocalEmbeddings(settings)
    documents = DocumentService(database, knowledge, embeddings, agents, settings)
    chat = ChatService(
        database, agents, documents, knowledge, embeddings, ProviderRegistry(settings), settings
    )
    return Services(database, knowledge, agents, documents, chat)
