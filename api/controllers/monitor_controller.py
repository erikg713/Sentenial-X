# api/controllers/monitor_controller.py

from fastapi import APIRouter
import psutil
import datetime
import socket

router = APIRouter(prefix="/monitor", tags=["Monitor"])


@router.get("/health")
async def health_check():
    """
    ✅ Basic health check for the API service
    """
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hostname": socket.gethostname(),
    }


@router.get("/system")
async def system_stats():
    """
    ✅ Get CPU, memory, and disk usage
    """
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "swap": psutil.swap_memory()._asdict(),
        "disk": {part.mountpoint: psutil.disk_usage(part.mountpoint)._asdict()
                 for part in psutil.disk_partitions()},
    }


@router.get("/network")
async def network_stats():
    """
    ✅ Get real-time network statistics
    """
    io_counters = psutil.net_io_counters(pernic=True)
    return {
        nic: {
            "bytes_sent": counters.bytes_sent,
            "bytes_recv": counters.bytes_recv,
            "packets_sent": counters.packets_sent,
            "packets_recv": counters.packets_recv,
        }
        for nic, counters in io_counters.items()
    }


@router.get("/processes")
async def list_processes(limit: int = 10):
    """
    ✅ List top processes by memory usage
    """
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        processes.append(proc.info)

    processes = sorted(processes, key=lambda x: x["memory_percent"], reverse=True)[:limit]
    return {"top_processes": processes} 
