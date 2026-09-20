import uuid

from app.modules.face import run_face, verify_batch_faces
from app.modules.ocr import _PASSPORT_OUTPUT_KEYS, classify_document, run_ocr
from app.modules.tampering import run_tampering
from app.modules.validation import cross_validate, validate
from app.schemas import (
    BatchScreeningResult,
    CrossDocumentCheck,
    DocumentType,
    Finding,
    Recommendation,
    RiskLevel,
    ScreeningResult,
    TravellerDossier,
)
from app.scoring import aggregate, aggregate_batch


def screen(
    document_type: DocumentType,
    document_bytes: bytes,
    live_bytes: bytes | None = None,
    text_hint: str | None = None,
    filename: str | None = None,
    document_id: str | None = None,
) -> ScreeningResult:
    ocr, ocr_findings = run_ocr(document_type, document_bytes, text_hint=text_hint)
    validation, val_findings = validate(document_type, ocr.details)
    tampering, tamp_findings = run_tampering(document_bytes)
    face, face_findings = run_face(document_bytes, live_bytes, document_type.value)

    if document_type == DocumentType.PASSPORT:
        extracted = {
            k: ocr.details[k]
            for k in _PASSPORT_OUTPUT_KEYS
            if k in ocr.details and ocr.details[k] not in (None, "", {})
        }
    else:
        extracted = {k: v for k, v in ocr.details.items() if k not in {"raw_text", "image_size"}}
    result = ScreeningResult(
        document_type=document_type,
        authenticity_score=0,
        risk_level=RiskLevel.HIGH,
        recommendation=Recommendation.REFER_FRAUD_CELL,
        ocr=ocr,
        validation=validation,
        tampering=tampering,
        face=face,
        findings=ocr_findings + val_findings + tamp_findings + face_findings,
        extracted_fields=extracted,
        filename=filename,
        document_id=document_id,
    )
    return aggregate(result)


def screen_batch(
    documents: list[dict],
    live_bytes: bytes | None = None,
) -> BatchScreeningResult:
    batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
    screened_docs: list[ScreeningResult] = []
    docs_for_cross: list[dict] = []
    docs_for_face: list[dict] = []
    all_findings: list[Finding] = []

    # 1. Screen each document individually
    for index, item in enumerate(documents, start=1):
        doc_bytes = item.get("document_bytes") or item.get("data", b"")
        fname = item.get("filename") or f"document_{index}.jpg"
        doc_id = item.get("document_id") or f"DOC-{index:02d}"
        hint = item.get("text_hint")
        raw_type = item.get("document_type")

        # Determine document type (auto-detect if missing or GENERIC)
        if isinstance(raw_type, str):
            try:
                dtype = DocumentType(raw_type)
            except ValueError:
                dtype = DocumentType.GENERIC
        elif isinstance(raw_type, DocumentType):
            dtype = raw_type
        else:
            dtype = None

        if dtype is None or dtype == DocumentType.GENERIC:
            detected, _, _ = classify_document(doc_bytes, hint)
            dtype = detected

        single_result = screen(
            document_type=dtype,
            document_bytes=doc_bytes,
            live_bytes=live_bytes,
            text_hint=hint,
            filename=fname,
            document_id=doc_id,
        )
        screened_docs.append(single_result)
        all_findings.extend(single_result.findings)

        docs_for_cross.append({
            "doc_id": doc_id,
            "filename": fname,
            "doc_type": dtype.value,
            "fields": single_result.extracted_fields,
        })
        docs_for_face.append({
            "doc_id": doc_id,
            "filename": fname,
            "doc_type": dtype.value,
            "image_bytes": doc_bytes,
        })

    # 2. Multi-Document Face Analysis
    cross_face_matches, live_face_matches, face_findings = verify_batch_faces(
        docs_for_face, live_bytes=live_bytes
    )
    all_findings.extend(face_findings)

    # 3. Cross-Document Validation Checks
    cross_checks, cross_findings = cross_validate(docs_for_cross)
    all_findings.extend(cross_findings)

    # 4. Synthesize Consolidated Traveller Dossier
    dossier = _build_traveller_dossier(docs_for_cross, cross_checks)

    # 5. Build Initial Batch Result
    batch_result = BatchScreeningResult(
        batch_id=batch_id,
        overall_authenticity_score=0.0,
        risk_level=RiskLevel.HIGH,
        recommendation=Recommendation.REFER_FRAUD_CELL,
        dossier=dossier,
        cross_checks=cross_checks,
        documents=screened_docs,
        cross_face_matches=cross_face_matches,
        live_face_matches=live_face_matches,
        total_findings=all_findings,
        summary="",
    )

    # 6. Holistic Multi-Document Scoring
    return aggregate_batch(batch_result)


def _build_traveller_dossier(
    docs_for_cross: list[dict], cross_checks: list[CrossDocumentCheck]
) -> TravellerDossier:
    primary_name = None
    primary_dob = None
    primary_gender = None
    primary_nationality = None
    passport_number = None
    visa_number = None
    aadhaar_number = None
    dl_number = None
    addresses: list[str] = []
    presented: list[str] = []

    # Priority order for primary identity: Passport > Aadhaar > DL > Visa
    priority_types = ["indian_passport", "national_id", "driving_license", "visa", "generic_document"]
    docs_sorted = sorted(
        docs_for_cross,
        key=lambda d: priority_types.index(d["doc_type"]) if d["doc_type"] in priority_types else 99,
    )

    for d in docs_sorted:
        f = d.get("fields", {})
        dtype = d.get("doc_type")
        fname = d.get("filename", "")

        # Names
        name = f.get("full_name") or (
            f"{f.get('given_names', '')} {f.get('surname', '')}".strip()
            if f.get("given_names") or f.get("surname")
            else None
        )
        if name:
            if not primary_name or len(name.split()) > len(primary_name.split()):
                primary_name = name

        # DOB
        dob = f.get("date_of_birth")
        if not primary_dob and dob:
            primary_dob = dob

        # Gender
        gender = f.get("sex")
        if not primary_gender and gender:
            primary_gender = gender

        # Nationality
        nat = f.get("nationality") or ("IND" if f.get("issuer_india") else None)
        if not primary_nationality and nat:
            primary_nationality = nat

        # Specific credentials
        if dtype == "indian_passport" and f.get("passport_number"):
            passport_number = f.get("passport_number")
            presented.append(f"Passport ({passport_number})")
        elif dtype == "visa" and f.get("visa_number"):
            visa_number = f.get("visa_number")
            presented.append(f"Visa ({visa_number})")
        elif dtype == "national_id" and (f.get("aadhaar_display") or f.get("aadhaar_number")):
            aadhaar_number = f.get("aadhaar_display") or f.get("aadhaar_number")
            presented.append(f"Aadhaar ({aadhaar_number})")
        elif dtype == "driving_license" and f.get("dl_number"):
            dl_number = f.get("dl_number")
            presented.append(f"Driving Licence ({dl_number})")
        else:
            doc_label = dtype.replace("_", " ").title()
            presented.append(f"{doc_label} ({fname})")

        # Addresses
        addr = f.get("address") or f.get("place_of_birth")
        if addr and addr not in addresses:
            addresses.append(addr)

    # Check if there are discrepancies
    has_mismatch = any(c.status == "MISMATCH" and c.severity == "high" for c in cross_checks)
    has_partial = any(c.status == "PARTIAL" or c.severity == "medium" for c in cross_checks)
    if has_mismatch:
        consistency = "DISCREPANCY"
    elif has_partial:
        consistency = "PARTIAL_MATCH"
    else:
        consistency = "CONSISTENT"

    return TravellerDossier(
        primary_name=primary_name,
        primary_dob=primary_dob,
        primary_gender=primary_gender,
        primary_nationality=primary_nationality,
        passport_number=passport_number,
        visa_number=visa_number,
        aadhaar_number=aadhaar_number,
        dl_number=dl_number,
        addresses=addresses,
        documents_presented=presented,
        identity_consistency=consistency,
    )

