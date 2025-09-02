# api/controllers/reports_controller.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import datetime
import io

from api.database import get_db
from api.models import IncidentLog
from api.schemas import ReportResponse, IncidentSchema
from api.utils.auth import get_current_user

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


@router.get("/", response_model=List[IncidentSchema])
def get_all_incidents(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    """
    Fetch all logged incidents from the system.
    """
    incidents = db.query(IncidentLog).order_by(IncidentLog.timestamp.desc()).all()
    return incidents


@router.get("/{incident_id}", response_model=IncidentSchema)
def get_single_incident(
    incident_id: int, 
    db: Session = Depends(get_db), 
    user: dict = Depends(get_current_user)
):
    """
    Fetch a single incident by ID.
    """
    incident = db.query(IncidentLog).filter(IncidentLog.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@router.get("/generate/daily", response_model=ReportResponse)
def generate_daily_report(
    db: Session = Depends(get_db), 
    user: dict = Depends(get_current_user)
):
    """
    Generate a daily incident summary report.
    """
    today = datetime.date.today()
    start = datetime.datetime.combine(today, datetime.time.min)
    end = datetime.datetime.combine(today, datetime.time.max)

    incidents = (
        db.query(IncidentLog)
        .filter(IncidentLog.timestamp >= start, IncidentLog.timestamp <= end)
        .all()
    )

    summary = f"Daily Report ({today})\n"
    summary += f"Total Incidents: {len(incidents)}\n"
    summary += "-" * 40 + "\n"

    for i in incidents:
        summary += f"[{i.timestamp}] {i.type.upper()} - {i.severity} - {i.description}\n"

    return ReportResponse(
        report_title="Daily Threat Report",
        report_date=today,
        total_incidents=len(incidents),
        content=summary
    )


@router.get("/generate/custom", response_model=ReportResponse)
def generate_custom_report(
    start_date: datetime.date,
    end_date: datetime.date,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    """
    Generate a custom incident report for a given date range.
    """
    start = datetime.datetime.combine(start_date, datetime.time.min)
    end = datetime.datetime.combine(end_date, datetime.time.max)

    incidents = (
        db.query(IncidentLog)
        .filter(IncidentLog.timestamp >= start, IncidentLog.timestamp <= end)
        .all()
    )

    summary = f"Custom Report ({start_date} to {end_date})\n"
    summary += f"Total Incidents: {len(incidents)}\n"
    summary += "-" * 40 + "\n"

    for i in incidents:
        summary += f"[{i.timestamp}] {i.type.upper()} - {i.severity} - {i.description}\n"

    return ReportResponse(
        report_title="Custom Threat Report",
        report_date=datetime.date.today(),
        total_incidents=len(incidents),
        content=summary
    )
