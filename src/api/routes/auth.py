"""Authentication endpoints for user and admin login."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas.auth import (
    AdminLoginRequest,
    AdminLoginResponse,
    UserLoginRequest,
    UserLoginResponse,
    UserRegisterRequest,
    UserRegisterResponse,
)
from src.service.auth_service import login_admin, login_user, register_user


router = APIRouter()


@router.post("/auth/register", response_model=UserRegisterResponse)
async def register(request: UserRegisterRequest) -> UserRegisterResponse:
    """Register user with name, student_id, phone and password."""
    try:
        result = register_user(
            name=request.name,
            student_id=request.student_id,
            phone=request.phone,
            password=request.password,
        )
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"register service error: {service_error}") from service_error

    return UserRegisterResponse(**result)


@router.post("/auth/login", response_model=UserLoginResponse)
async def user_login(request: UserLoginRequest) -> UserLoginResponse:
    """User login by account (student_id or phone) and password."""
    try:
        result = login_user(account=request.account, password=request.password)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"user login service error: {service_error}") from service_error

    return UserLoginResponse(**result)


@router.post("/auth/admin/login", response_model=AdminLoginResponse)
async def admin_login(request: AdminLoginRequest) -> AdminLoginResponse:
    """Admin login by fixed account and password."""
    try:
        result = login_admin(username=request.username, password=request.password)
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error)) from value_error
    except Exception as service_error:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"admin login service error: {service_error}") from service_error

    return AdminLoginResponse(**result)
