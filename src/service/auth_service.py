"""Authentication service for user registration and login."""

from __future__ import annotations

from typing import Any

from src.database.db_operate import (
    add_user_auth,
    ensure_user_auth_schema,
    get_user_auth_by_account_and_password,
    get_user_auth_by_phone,
    get_user_auth_by_student_id,
    get_user_auth_for_login,
)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123456"


def _normalize_text(value: str | None, field_name: str, max_length: int) -> str:
    """Normalize and validate plain text fields."""
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} length must be <= {max_length}")
    return normalized


def _build_user_payload(user_record: dict[str, Any]) -> dict[str, Any]:
    """Build standardized user payload for responses."""
    return {
        "id": user_record["id"],
        "name": user_record["name"],
        "student_id": user_record["student_id"],
        "phone": user_record["phone"],
        "created_at": user_record["created_at"],
    }


def register_user(name: str, student_id: str, phone: str, password: str) -> dict[str, Any]:
    """Register one user by name, student id, phone and password."""
    normalized_name = _normalize_text(name, "name", 50)
    normalized_student_id = _normalize_text(student_id, "student_id", 30)
    normalized_phone = _normalize_text(phone, "phone", 20)
    normalized_password = _normalize_text(password, "password", 50)

    ensure_user_auth_schema()
    existing_user = get_user_auth_by_student_id(normalized_student_id)
    if existing_user:
        raise ValueError("student_id already registered")

    existing_phone = get_user_auth_by_phone(normalized_phone)
    if existing_phone:
        raise ValueError("phone already registered")

    success = add_user_auth(
        name=normalized_name,
        student_id=normalized_student_id,
        phone=normalized_phone,
        password=normalized_password,
    )
    if not success:
        raise RuntimeError("failed to register user")

    created_user = get_user_auth_by_student_id(normalized_student_id)
    if not created_user:
        raise RuntimeError("failed to load registered user")

    return {
        "success": True,
        "message": "register success",
        "user": _build_user_payload(created_user),
    }


def login_user(account: str, password: str) -> dict[str, Any]:
    """Login user by account (student_id or phone) and password."""
    normalized_account = _normalize_text(account, "account", 30)
    normalized_password = _normalize_text(password, "password", 50)

    ensure_user_auth_schema()
    user_record = get_user_auth_by_account_and_password(normalized_account, normalized_password)
    if not user_record:
        raise ValueError("invalid account or password")

    return {
        "success": True,
        "message": "login success",
        "user": _build_user_payload(user_record),
    }


def login_admin(username: str, password: str) -> dict[str, Any]:
    """Login admin with fixed account and password."""
    normalized_username = _normalize_text(username, "username", 50)
    normalized_password = _normalize_text(password, "password", 50)

    if normalized_username != ADMIN_USERNAME or normalized_password != ADMIN_PASSWORD:
        raise ValueError("invalid admin username or password")

    return {
        "success": True,
        "message": "admin login success",
        "role": "admin",
    }
