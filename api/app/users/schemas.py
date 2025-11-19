from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.users.models import UserRole

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.CLIENT

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserInDB(UserBase):
    id: int
    hashed_password: str
    is_active: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True