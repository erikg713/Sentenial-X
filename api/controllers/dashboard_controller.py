# api/controllers/dashboard_controller.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.database import get_db
from api.models import Threat, ExploitLog
import psutil
import datetime

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/system-stats")
def get_system_stats():
    """Return live system performance stats."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu": f"{cpu_percent}%",
            "memory": {
                "total": f"{round(memory.total / (1024**3), 2)} GB",
                "used": f"{round(memory.used / (1024**3), 2)} GB",
                "percent": f"{memory.percent}%"
            },
            "disk": {
                "total": f"{round(disk.total / (1024**3), 2)} GB",
                "used": f"{round(disk.used / (1024**3), 2)} GB",
                "percent": f"{disk.percent}%"
            },
            "uptime": str(datetime.timedelta(seconds=int(psutil.boot_time())))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threat-summary")
def get_threat_summary(db: Session = Depends(get_db)):
    """Return count of active and resolved threats."""
    try:
        total_threats = db.query(Threat).count()
        active_threats = db.query(Threat).filter(Threat.status == "active").count()
        resolved_threats = db.query(Threat).filter(Threat.status == "resolved").count()

        return {
            "total_threats": total_threats,
            "active_threats": active_threats,
            "resolved_threats": resolved_threats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exploit-logs")
def get_exploit_logs(limit: int = 10, db: Session = Depends(get_db)):
    """Return the latest exploit logs."""
    try:
        logs = (
            db.query(ExploitLog)
            .order_by(ExploitLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [{"id": log.id, "name": log.name, "status": log.status, "timestamp": log.timestamp} for log in logs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
