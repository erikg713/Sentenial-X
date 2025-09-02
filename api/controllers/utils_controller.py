# api/controllers/utils_controller.py

from fastapi import APIRouter, Depends
import platform
import psutil
import datetime

from api.utils.auth import get_current_user
from api.schemas import HealthResponse, SystemInfoResponse

router = APIRouter(
    prefix="/utils",
    tags=["Utilities"],
)


@router.get("/health", response_model=HealthResponse)
def health_check(user: dict = Depends(get_current_user)):
    """
    Simple health check endpoint.
    Returns status of API service.
    """
    return HealthResponse(
        status="ok",
        service="Sentenial-X API",
    )


@router.get("/system_info", response_model=SystemInfoResponse)
def system_info(user: dict = Depends(get_current_user)):
    """
    Returns detailed system information:
    CPU, memory, disk, network, OS details.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    return SystemInfoResponse(
        os=platform.system(),
        os_version=platform.version(),
        cpu_percent=cpu_percent,
        memory_total=memory.total,
        memory_used=memory.used,
        memory_percent=memory.percent,
        disk_total=disk.total,
        disk_used=disk.used,
        disk_percent=disk.percent,
        net_bytes_sent=net.bytes_sent,
        net_bytes_recv=net.bytes_recv,
        timestamp=datetime.datetime.utcnow(),
    )


@router.get("/timestamp")
def current_timestamp(user: dict = Depends(get_current_user)):
    """
    Returns the current UTC timestamp.
    """
    return {"timestamp": datetime.datetime.utcnow().isoformat()}


@router.get("/ping")
def ping():
    """
    Lightweight ping endpoint for connectivity check.
    """
    return {"status": "pong"}


@router.get("/uptime")
def uptime():
    """
    Returns API uptime since process start.
    """
    import time
    from api.config import START_TIME

    uptime_seconds = time.time() - START_TIME
    return {"uptime_seconds": uptime_seconds}
