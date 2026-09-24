from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from research_ai.database.models import Base


def _database_url() -> str:
    value = os.getenv("DATABASE_URL", "").strip()
    if value:
        return value
    data_root = Path(os.getenv("DATA_ROOT", "data"))
    data_root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_root / 'researchai.sqlite3'}"


DATABASE_URL = _database_url()
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def create_db_and_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
