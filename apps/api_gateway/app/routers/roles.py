# app/routers/roles.py
from fastapi import APIRouter, Depends, HTTPException, status
from ..rbac import set_role, get_role, delete_role, list_roles
from ..deps import require_scope
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/roles", tags=["roles"])

class RolePayload(BaseModel):
    name: str
    scopes: List[str]

# Admin-only endpoints:
admin_required = Depends(require_scope("admin"))

@router.post("/", dependencies=[admin_required])
async def create_or_update_role(payload: RolePayload):
    await set_role(payload.name, payload.scopes)
    return {"ok": True, "role": payload.name, "scopes": payload.scopes}

@router.get("/", dependencies=[admin_required])
async def get_all_roles():
    return await list_roles()

@router.get("/{role_name}", dependencies=[admin_required])
async def get_role_route(role_name: str):
    r = await get_role(role_name)
    if r is None:
        raise HTTPException(status_code=404, detail="role not found")
    return {"role": role_name, "scopes": r}

@router.delete("/{role_name}", dependencies=[admin_required])
async def delete_role_route(role_name: str):
    await delete_role(role_name)
    return {"ok": True}