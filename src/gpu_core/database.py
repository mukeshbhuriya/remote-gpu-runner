"""
Database engine, session management, and base model.

Uses SQLAlchemy 2.0+ patterns. Currently backed by SQLite;
designed so PostgreSQL can be swapped in by changing the URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from gpu_core.config import get_config
from gpu_core.logging_config import get_logger

logger = get_logger("database")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# Module-level engine and session factory (lazy-init)
_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def _enable_sqlite_wal(dbapi_conn, connection_record) -> None:  # type: ignore[no-untyped-def]
    """Enable WAL mode for SQLite for better concurrent read performance."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine(database_url: str | None = None):  # type: ignore[no-untyped-def]
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        return _engine

    if database_url is None:
        config = get_config()
        database_url = config.database.database_url

    # Ensure the directory for SQLite exists
    if database_url.startswith("sqlite"):
        db_path = database_url.replace("sqlite:///", "")
        if db_path.startswith("./"):
            db_path_obj = Path(db_path)
        else:
            db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        echo=False,
        pool_pre_ping=True,
    )

    # Enable WAL mode for SQLite
    if database_url.startswith("sqlite"):
        event.listen(_engine, "connect", _enable_sqlite_wal)

    logger.info(f"Database engine created: {database_url.split('://')[0]}")
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Get or create the session factory."""
    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory

    engine = get_engine(database_url)
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionFactory


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that yields a database session.

    Usage with FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(database_url: str | None = None) -> None:
    """
    Initialize the database — create all tables.

    In production, use Alembic migrations instead.
    """
    engine = get_engine(database_url)
    # Import models so they register with Base.metadata
    import gpu_core.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")


def reset_db_engine() -> None:
    """Reset the engine and session factory (for testing)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
