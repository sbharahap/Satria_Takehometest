import logging
from dataclasses import dataclass

from src.config import Settings
from src.database.connection import Database
from src.knowledge.vector_store import KnowledgeStore
from src.services.agents import AgentService


@dataclass
class Services:
    database: Database
    knowledge: KnowledgeStore
    agents: AgentService


def initialize(settings: Settings) -> Services:
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    settings.ensure_directories()
    database = Database(settings.sqlite_path)
    database.initialize()
    knowledge = KnowledgeStore(settings)
    return Services(database, knowledge, AgentService(database, knowledge))
