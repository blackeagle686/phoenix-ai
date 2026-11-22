"""PHX v2 Semantic-Structural Code Capsule Manifest Schema (Section 5.3)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CapsuleStatus(str, Enum):
    ACTIVE = "ACTIVE"
    QUARANTINED = "QUARANTINED"


class CapsuleStack(BaseModel):
    language: str = "python"
    framework: str = "stdlib"
    framework_semver: str = ">=0.0.0"
    orm: str = "none"


class CapsuleSemanticProfile(BaseModel):
    summary: str
    embedding_model: str = "phx-hash-24d"
    tags: List[str] = Field(default_factory=list)


class CapsuleContract(BaseModel):
    provides: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)
    middleware_order_priority: int = 100


class CapsuleParameter(BaseModel):
    name: str
    type: str = "str"  # int | float | str | bool | identifier
    default: Any = None
    constraints: Optional[str] = None  # e.g. "60..900" or "a|b|c"


class CapsuleASTWiring(BaseModel):
    template_path: Optional[str] = None
    template_source: str = ""
    injection_points: List[str] = Field(default_factory=list)
    custom_logic_slots: List[str] = Field(default_factory=list)


class CapsuleVerification(BaseModel):
    test_suite_path: Optional[str] = None
    test_source: str = ""
    security_audited: bool = False
    last_verified_at: Optional[str] = None
    status: CapsuleStatus = CapsuleStatus.ACTIVE
    quarantine_reason: Optional[str] = None


class CodeCapsule(BaseModel):
    """Contract-validated, security-audited, test-backed Code Capsule (Section 5.3)."""

    capsule_id: str
    version: str = "1.0.0"
    stack: CapsuleStack = Field(default_factory=CapsuleStack)
    semantic_profile: CapsuleSemanticProfile
    contract: CapsuleContract = Field(default_factory=CapsuleContract)
    parameters: List[CapsuleParameter] = Field(default_factory=list)
    ast_wiring: CapsuleASTWiring = Field(default_factory=CapsuleASTWiring)
    verification: CapsuleVerification = Field(default_factory=CapsuleVerification)
    is_mega_capsule: bool = False
    constituent_ids: List[str] = Field(default_factory=list)

    @field_validator("capsule_id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("capsule_id cannot be empty")
        return s
