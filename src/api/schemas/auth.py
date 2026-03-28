"""Schemas for authentication APIs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    """Request payload for user registration."""

    name: str = Field(..., min_length=1, max_length=50, description="User name")
    student_id: str = Field(..., min_length=1, max_length=30, description="Student id")
    phone: str = Field(..., min_length=6, max_length=20, description="Mobile phone")
    password: str = Field(..., min_length=6, max_length=50, description="Login password")


class UserLoginRequest(BaseModel):
    """Request payload for user login."""

    account: str = Field(..., min_length=1, max_length=30, description="Student id or phone")
    password: str = Field(..., min_length=1, max_length=50, description="Login password")


class AdminLoginRequest(BaseModel):
    """Request payload for admin login."""

    username: str = Field(..., min_length=1, max_length=50, description="Admin username")
    password: str = Field(..., min_length=1, max_length=50, description="Admin password")


class AuthUserInfo(BaseModel):
    """Common user info payload in auth responses."""

    id: int
    name: str
    student_id: str
    phone: str
    created_at: datetime


class UserRegisterResponse(BaseModel):
    """Response payload for user registration."""

    success: bool
    message: str
    user: AuthUserInfo


class UserLoginResponse(BaseModel):
    """Response payload for user login."""

    success: bool
    message: str
    user: AuthUserInfo


class AdminLoginResponse(BaseModel):
    """Response payload for admin login."""

    success: bool
    message: str
    role: str
