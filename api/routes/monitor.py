# routes/monitor.py
"""
Monitor Routes for Sentenial-X
Handles system monitoring, telemetry retrieval, and background scan orchestration.
"""

from fastapi import APIRouter, HTTPException
from utils.response import success_response, error_response
from services.monitor_service import MonitorService

router = APIRouter(prefix="/monitor", tags=["Monitor"])

# Initialize monitoring service
monitor_service = MonitorService()


@router.get("/status")
async def get_system_status():
    """
    Returns current monitoring status (CPU, memory, processes, network, alerts).
    """
    try:
        status = monitor_service.collect_system_status()
        return success_response("System status retrieved successfully", status)
    except Exception as e:
        return error_response(f"Failed to retrieve system status: {str(e)}", 500)


@router.post("/scan/start")
async def start_monitoring_scan(scan_type: str = "full"):
    """
    Starts a monitoring scan.
    scan_type can be 'full', 'network', 'process', or 'file_integrity'.
    """
    try:
        task_id = monitor_service.start_scan(scan_type)
        return success_response(f"Monitoring scan ({scan_type}) started", {"task_id": task_id})
    except ValueError as ve:
        return error_response(str(ve), 400)
    except Exception as e:
        return error_response(f"Failed to start monitoring scan: {str(e)}", 500)


@router.get("/scan/{task_id}")
async def get_scan_results(task_id: str):
    """
    Retrieve results of a completed scan by task_id.
    """
    try:
        results = monitor_service.get_scan_results(task_id)
        if not results:
            raise HTTPException(status_code=404, detail="Scan results not found or still processing")
        return success_response("Scan results retrieved", results)
    except HTTPException as he:
        raise he
    except Exception as e:
        return error_response(f"Failed to retrieve scan results: {str(e)}", 500)


@router.get("/alerts")
async def get_alerts():
    """
    Returns all active alerts logged by the monitor service.
    """
    try:
        alerts = monitor_service.get_alerts()
        return success_response("Active alerts retrieved", alerts)
    except Exception as e:
        return error_response(f"Failed to retrieve alerts: {str(e)}", 500)


@router.delete("/alerts/clear")
async def clear_alerts():
    """
    Clears all active alerts.
    """
    try:
        monitor_service.clear_alerts()
        return success_response("All alerts cleared")
    except Exception as e:
        return error_response(f"Failed to clear alerts: {str(e)}", 500) 