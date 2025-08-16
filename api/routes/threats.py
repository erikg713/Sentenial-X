from fastapi import APIRouter, Depends
from api.controllers.threats_controller import ThreatsController
from api.utils.auth import get_current_user

router = APIRouter()
controller = ThreatsController()

@router.get("/")
async def list_threats(user: dict = Depends(get_current_user)):
    return await controller.list_threats()

@router.post("/simulate")
async def simulate_threat(payload: dict, user: dict = Depends(get_current_user)):
    return await controller.simulate_threat(payload) 