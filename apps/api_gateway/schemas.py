from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


class UserBase(BaseModel):
    username: str = Field(..., example="admin")
    email: Optional[EmailStr] = Field(None, example="admin@example.com")
    full_name: Optional[str] = Field(None, example="Admin User")
    disabled: Optional[bool] = Field(False, description="Is the user disabled?")


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, example="strongpassword123")


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class User(UserBase):
    id: int

    class Config:
        orm_mode = True
