"""
Semantic Service Layer
----------------------
Handles persistence of threats, telemetry, alerts, exploits into PostgreSQL.
"""

from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import Column, String, DateTime, JSON, Integer
from sqlalchemy.orm import declarative_base

from api.config import settings

Base = declarative_base()

# -------------------------------
# Database Models
# -------------------------------

class ThreatDB(Base):
    __tablename__ = "threats"
    id = Column(String, primary_key=True)
    severity = Column(String, nullable=False)
    description = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)


class TelemetryDB(Base):
    __tablename__ = "telemetry"
    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    data = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AlertDB(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True)
    severity = Column(String, nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, default="triggered")
    timestamp = Column(DateTime, default=datetime.utcnow)


class ExploitDB(Base):
    __tablename__ = "exploits"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    cve = Column(String, nullable=True)
    description = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)


# -------------------------------
# Database Session
# -------------------------------
engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db():
    """Initialize database tables (Alembic recommended in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# -------------------------------
# CRUD Operations
# -------------------------------
class SemanticService:
    @staticmethod
    async def save_threat(threat: dict):
        async with SessionLocal() as session:
            obj = ThreatDB(**threat)
            session.add(obj)
            await session.commit()
            return obj

    @staticmethod
    async def save_telemetry(event: dict):
        async with SessionLocal() as session:
            obj = TelemetryDB(**event)
            session.add(obj)
            await session.commit()
            return obj

    @staticmethod
    async def save_alert(alert: dict):
        async with SessionLocal() as session:
            obj = AlertDB(**alert)
            session.add(obj)
            await session.commit()
            return obj

    @staticmethod
    async def save_exploit(exploit: dict):
        async with SessionLocal() as session:
            obj = ExploitDB(**exploit)
            session.add(obj)
            await session.commit()
            return obj

    @staticmethod
    async def list_threats():
        async with SessionLocal() as session:
            return (await session.execute("SELECT * FROM threats")).all()

    @staticmethod
    async def list_alerts():
        async with SessionLocal() as session:
            return (await session.execute("SELECT * FROM alerts")).all()
