from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.models.models import Token, User, UserCreate
from api.security.auth import (
    authenticate_user, create_access_token, get_current_active_user, 
    ACCESS_TOKEN_EXPIRE_MINUTES, create_user
)

router = APIRouter(tags=["authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    from api.security.auth import fake_users_db
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/users", response_model=User, status_code=201)
async def register_user(user: UserCreate):
    """
    Create a new user.
    """
    success = create_user(
        username=user.username,
        email=user.email,
        password=user.password,
        full_name=user.full_name
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Return the user without sensitive information
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=False
    )


@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user