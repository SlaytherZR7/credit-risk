from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from datetime import timedelta

from app.users.schemas import UserCreate
from app.users.repository import UserRepository
from app.auth.utils import verify_password, get_password_hash, create_access_token
from app.auth.schemas import LoginResponse
from app.users.schemas import UserResponse
from app.config import settings

class AuthService:
    
    @staticmethod
    def register_user(db: Session, user_data: UserCreate) -> UserResponse:
        
        if UserRepository.get_by_email(db, email=user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The email is already registered"
            )
        
        hashed_password = get_password_hash(user_data.password)
        
        user = UserRepository.create(db, user_data, hashed_password)
        
        return UserResponse.model_validate(user)
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> LoginResponse:

        user = UserRepository.get_by_email(db, email=email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not UserRepository.is_active(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is inactive"
            )
        
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": user.role.value},
            expires_delta=access_token_expires
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )