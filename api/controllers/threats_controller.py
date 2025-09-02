# api/controllers/threats_controller.py

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from api.database import get_db
from api.models import Threat
from api.schemas import ThreatCreate, ThreatResponse

router = APIRouter(
    prefix="/threats",
    tags=["Threats"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=ThreatResponse)
def create_threat(threat: ThreatCreate, db: Session = get_db()):
    """
    Create a new threat entry in the database.
    """
    db_threat = Threat(
        name=threat.name,
        severity=threat.severity,
        description=threat.description,
        status=threat.status,
        source=threat.source,
    )
    db.add(db_threat)
    db.commit()
    db.refresh(db_threat)
    return db_threat


@router.get("/", response_model=List[ThreatResponse])
def get_threats(
    db: Session = get_db(),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    source: Optional[str] = Query(None, description="Filter by source"),
):
    """
    Retrieve all threats with optional filters for severity, status, and source.
    """
    query = db.query(Threat)

    if severity:
        query = query.filter(Threat.severity == severity)
    if status:
        query = query.filter(Threat.status == status)
    if source:
        query = query.filter(Threat.source == source)

    threats = query.all()
    return threats


@router.get("/{threat_id}", response_model=ThreatResponse)
def get_threat(threat_id: int, db: Session = get_db()):
    """
    Retrieve a specific threat by ID.
    """
    threat = db.query(Threat).filter(Threat.id == threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    return threat


@router.put("/{threat_id}", response_model=ThreatResponse)
def update_threat(threat_id: int, threat_data: ThreatCreate, db: Session = get_db()):
    """
    Update an existing threat entry.
    """
    threat = db.query(Threat).filter(Threat.id == threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")

    threat.name = threat_data.name
    threat.severity = threat_data.severity
    threat.description = threat_data.description
    threat.status = threat_data.status
    threat.source = threat_data.source

    db.commit()
    db.refresh(threat)
    return threat


@router.delete("/{threat_id}")
def delete_threat(threat_id: int, db: Session = get_db()):
    """
    Delete a threat from the database.
    """
    threat = db.query(Threat).filter(Threat.id == threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")

    db.delete(threat)
    db.commit()
    return {"detail": f"Threat with ID {threat_id} deleted successfully"}
