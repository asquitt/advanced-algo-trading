"""
Database connection and session management.

This module provides SQLAlchemy setup for PostgreSQL and connection pooling.
Uses async SQLAlchemy for better performance with FastAPI.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from src.utils.config import settings
from src.utils.logger import app_logger

# SQLAlchemy metadata and base
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Create engine with connection pooling for efficiency
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Additional connections if pool is full
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


@contextmanager
def get_db():
    """
    Context manager for database sessions.

    Usage:
        with get_db() as db:
            result = db.query(TradingSignal).all()

    Automatically handles:
    - Session creation
    - Commit on success
    - Rollback on error
    - Session cleanup
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        app_logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.

    This creates all tables defined in our models.
    In production, use Alembic migrations instead.
    """
    try:
        Base.metadata.create_all(bind=engine)
        app_logger.info("Database tables initialized successfully")
    except Exception as e:
        app_logger.error(f"Failed to initialize database: {e}")
        raise


def get_db_dependency():
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI routes:
        @app.get("/signals")
        def get_signals(db: Session = Depends(get_db_dependency)):
            return db.query(TradingSignal).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
