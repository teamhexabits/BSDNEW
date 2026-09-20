from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PASSPORT = "indian_passport"
    VISA = "visa"
    NATIONAL_ID = "national_id"
    DRIVING_LICENSE = "driving_license"
    GENERIC = "generic_document"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(str, Enum):
    ALLOW = "ALLOW"
    SECONDARY_INSPECTION = "SECONDARY_INSPECTION"
    REFER_FRAUD_CELL = "REFER_FRAUD_CELL"


class Finding(BaseModel):
    module: str
    severity: str
    code: str
    message: str


class ModuleResult(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class FaceVerifyResponse(BaseModel):
    matched: bool
    similarity: float | None
    threshold: float
    confidence: float
    message: str
    reason: str | None = None


class ScreeningResult(BaseModel):
    document_type: DocumentType
    authenticity_score: float
    risk_level: RiskLevel
    recommendation: Recommendation
    ocr: ModuleResult
    validation: ModuleResult
    tampering: ModuleResult
    face: ModuleResult
    findings: list[Finding]
    extracted_fields: dict[str, Any]
    filename: str | None = None
    document_id: str | None = None


class CrossDocumentCheck(BaseModel):
    name: str
    status: str  # MATCH, MISMATCH, PARTIAL, WARNING, INFO
    summary: str
    severity: str  # low, medium, high, info
    details: dict[str, Any] = Field(default_factory=dict)


class TravellerDossier(BaseModel):
    primary_name: str | None = None
    primary_dob: str | None = None
    primary_gender: str | None = None
    primary_nationality: str | None = None
    passport_number: str | None = None
    visa_number: str | None = None
    aadhaar_number: str | None = None
    dl_number: str | None = None
    addresses: list[str] = Field(default_factory=list)
    documents_presented: list[str] = Field(default_factory=list)
    identity_consistency: str = "CONSISTENT"  # CONSISTENT, DISCREPANCY, UNVERIFIED


class CrossDocFaceMatch(BaseModel):
    doc1_id: str
    doc1_type: str
    doc2_id: str
    doc2_type: str
    similarity: float | None
    matched: bool
    threshold: float = 0.40


class BatchScreeningResult(BaseModel):
    batch_id: str
    overall_authenticity_score: float
    risk_level: RiskLevel
    recommendation: Recommendation
    dossier: TravellerDossier
    cross_checks: list[CrossDocumentCheck]
    documents: list[ScreeningResult]
    cross_face_matches: list[CrossDocFaceMatch] = Field(default_factory=list)
    live_face_matches: list[dict[str, Any]] = Field(default_factory=list)
    total_findings: list[Finding]
    summary: str

