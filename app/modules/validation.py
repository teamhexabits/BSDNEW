"""Document validation: formats, checksums, dates, and cross-field consistency."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from app.documents import patterns as P
from app.schemas import DocumentType, Finding, ModuleResult

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CT", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _add(findings: list[Finding], severity: str, code: str, message: str) -> None:
    findings.append(Finding(module="validation", severity=severity, code=code, message=message))


def validate(document_type: DocumentType, fields: dict) -> tuple[ModuleResult, list[Finding]]:
    findings: list[Finding] = []
    checks: dict[str, bool] = {}

    if document_type == DocumentType.PASSPORT:
        number = str(fields.get("passport_number") or "")
        checks["passport_format"] = P.passport_number_ok(number)
        if not checks["passport_format"]:
            _add(
                findings,
                "high",
                "PASSPORT_FORMAT",
                "Passport number must be 1 letter + 7 digits (e.g. A1234567) or 2 letters + 6 digits (e.g. AB123456).",
            )

        line2 = fields.get("mrz_line2")
        if line2 and len(str(line2)) == 44:
            checks["mrz_passport_cd"] = P.mrz_check_digit(line2[0:9], line2[9])
            checks["mrz_dob_cd"] = P.mrz_check_digit(line2[13:19], line2[19])
            checks["mrz_expiry_cd"] = P.mrz_check_digit(line2[21:27], line2[27])
            checks["mrz_composite"] = P.mrz_check_digit(
                line2[0:10] + line2[13:20] + line2[21:43], line2[43]
            )
            mrz_failures = [key for key, ok in checks.items() if key.startswith("mrz_") and not ok]
            if len(mrz_failures) >= 2:
                _add(
                    findings,
                    "high",
                    "MRZ_MULTIPLE_CHECK_FAIL",
                    "Multiple independent MRZ checks failed and should be reviewed as a possible non-genuine document.",
                )
            elif len(mrz_failures) == 1:
                _add(
                    findings,
                    "medium",
                    mrz_failures[0].upper(),
                    "MRZ check digit validation failed; the MRZ may contain an OCR error and should be reviewed.",
                )
            issuing = fields.get("issuing_country")
            nationality = fields.get("nationality")
            checks["india_codes"] = P.india_code_ok(issuing) and P.india_code_ok(nationality)
            if issuing and not P.india_code_ok(issuing):
                _add(findings, "medium", "ISSUER_NOT_IND", f"MRZ issuing state is {issuing}, not IND.")
        elif fields.get("mrz_present") and line2 and len(str(line2)) != 44:
            checks["mrz_length"] = False
            _add(
                findings,
                "medium",
                "MRZ_OCR_INCOMPLETE",
                "MRZ was detected but is not a complete 44-character TD3 line; review for OCR error before treating as forged.",
            )
        elif not fields.get("mrz_present"):
            _add(findings, "medium", "MRZ_MISSING", "Machine-readable zone not detected on the passport image.")
            checks["mrz_present"] = False

        issue = P.parse_date(fields.get("date_of_issue"))
        expiry_passport = P.parse_date(fields.get("date_of_expiry"))
        if issue and expiry_passport and expiry_passport <= issue:
            checks["issue_before_expiry"] = False
            _add(findings, "high", "DATE_ORDER", "Date of expiry is not after date of issue.")


    elif document_type == DocumentType.DRIVING_LICENSE:
        compact = re.sub(r"[^A-Z0-9]", "", str(fields.get("dl_number") or "").upper())
        checks["dl_format"] = P.dl_number_ok(compact)
        state = (fields.get("issuing_state") or compact[:2]).upper()
        checks["dl_state"] = state in INDIAN_STATES
        if not checks["dl_format"]:
            _add(
                findings,
                "high",
                "DL_FORMAT",
                "Driving licence number must be state code + RTO + year + serial (e.g. MH12 20260012345).",
            )
        elif not checks["dl_state"]:
            _add(findings, "medium", "DL_STATE", f"Issuing state code '{state}' is not a recognised Indian code.")

    elif document_type == DocumentType.NATIONAL_ID:
        aadhaar = str(fields.get("aadhaar_number") or "")
        if P.is_masked_aadhaar(aadhaar) or fields.get("masked"):
            digits = [c for c in aadhaar if c.isdigit()]
            checks["aadhaar_masked_format"] = len(digits) == 4
            checks["aadhaar_verhoeff"] = True  # Masked Aadhaar is a valid privacy format
            if not checks["aadhaar_masked_format"]:
                _add(findings, "medium", "AADHAAR_MASKED_INCOMPLETE", "Masked Aadhaar must show the final 4 digits.")
        else:
            digits = "".join(ch for ch in aadhaar if ch.isdigit())
            checks["aadhaar_length"] = len(digits) == 12
            checks["aadhaar_verhoeff"] = P.verhoeff_ok(digits) if digits else False
            if not checks["aadhaar_verhoeff"]:
                _add(
                    findings,
                    "high",
                    "AADHAAR_CHECKSUM",
                    "Aadhaar number failed the Verhoeff checksum used by UIDAI.",
                )

    elif document_type == DocumentType.GENERIC:
        checks["has_content"] = bool(fields.get("raw_text"))

    elif document_type == DocumentType.VISA:
        visa_no = str(fields.get("visa_number") or "")
        clean_vno = re.sub(r"[^A-Z0-9]", "", visa_no.upper())
        checks["visa_present"] = len(clean_vno) >= 6
        if not checks["visa_present"]:
            _add(findings, "high", "VISA_NUMBER_MISSING", "Visa number was not extracted or is too short.")
        if fields.get("passport_number"):
            checks["passport_link_present"] = P.passport_number_ok(str(fields["passport_number"]))

    expiry = P.parse_date(fields.get("date_of_expiry") or fields.get("valid_until"))
    if expiry:
        checks["not_expired"] = expiry >= _now()
        if not checks["not_expired"]:
            _add(findings, "high", "EXPIRED", "Document appears expired at time of screening.")

    dob = P.parse_date(fields.get("date_of_birth"))
    if dob:
        age = (_now() - dob).days / 365.25
        checks["plausible_age"] = 1 <= age <= 110
        if not checks["plausible_age"]:
            _add(findings, "high", "DOB_IMPLAUSIBLE", "Date of birth is outside a plausible human range.")
        if expiry and dob and expiry <= dob:
            checks["expiry_after_dob"] = False
            _add(findings, "high", "DATE_ORDER", "Expiry is not after date of birth.")

    valid_from = P.parse_date(fields.get("valid_from"))
    valid_until = P.parse_date(fields.get("valid_until"))
    if valid_from and valid_until:
        checks["visa_window"] = valid_until >= valid_from
        if not checks["visa_window"]:
            _add(findings, "high", "VISA_WINDOW", "Visa valid-until is earlier than valid-from.")

    passed = sum(1 for v in checks.values() if v)
    total = max(len(checks), 1)
    score = 100.0 * passed / total
    if not checks:
        score = 40.0
        _add(findings, "medium", "INSUFFICIENT_FIELDS", "Too few fields to complete a full format check.")

    high = sum(1 for f in findings if f.severity == "high")
    if high:
        score = min(score, 55)

    summary = f"{passed}/{total} structural checks passed."
    result = ModuleResult(
        name="validation",
        score=round(score, 1),
        confidence=85.0 if checks else 40.0,
        summary=summary,
        details={"checks": checks},
    )
    return result, findings


def cross_validate(
    documents: list[dict[str, Any]],
) -> tuple[list[CrossDocumentCheck], list[Finding]]:
    """Cross-document verification engine comparing extracted data across all uploaded documents."""
    from app.schemas import CrossDocumentCheck

    checks: list[CrossDocumentCheck] = []
    findings: list[Finding] = []

    # Map of docs by type
    docs_by_type: dict[str, list[dict[str, Any]]] = {}
    for doc in documents:
        dtype = str(doc.get("doc_type") or doc.get("document_type") or "")
        docs_by_type.setdefault(dtype, []).append(doc)

    # 1. NAME CONSISTENCY
    name_records: list[dict[str, Any]] = []
    for doc in documents:
        f = doc.get("fields", {})
        doc_label = doc.get("filename") or doc.get("doc_id") or doc.get("doc_type") or "Doc"
        dtype = doc.get("doc_type") or "Document"
        name = f.get("full_name") or (
            f"{f.get('given_names', '')} {f.get('surname', '')}".strip()
            if f.get("given_names") or f.get("surname")
            else None
        )
        if name and len(name) >= 3:
            name_records.append({
                "label": doc_label,
                "dtype": str(dtype),
                "name": name,
                "surname": f.get("surname"),
                "given_names": f.get("given_names"),
            })

    if len(name_records) >= 2:
        pairs_scores = []
        for i in range(len(name_records)):
            for j in range(i + 1, len(name_records)):
                rec_i = name_records[i]
                rec_j = name_records[j]
                sim = P.name_similarity(rec_i["name"], rec_j["name"])
                # Also check cross-field surname / given_names matches if one doc only has surname or given name
                if sim < 0.85:
                    alt_sims = [sim]
                    for part_i in (rec_i.get("surname"), rec_i.get("given_names")):
                        if part_i:
                            alt_sims.append(P.name_similarity(part_i, rec_j["name"]))
                    for part_j in (rec_j.get("surname"), rec_j.get("given_names")):
                        if part_j:
                            alt_sims.append(P.name_similarity(rec_i["name"], part_j))
                    if rec_i.get("surname") and rec_j.get("surname"):
                        alt_sims.append(P.name_similarity(rec_i["surname"], rec_j["surname"]))

                    # If one document only has the surname and another only has the first name,
                    # check if both are verified sub-components of a common parent full-name document in the dossier
                    for k in range(len(name_records)):
                        if k != i and k != j:
                            parent = name_records[k]
                            sim_i_k = P.name_similarity(rec_i["name"], parent["name"])
                            sim_j_k = P.name_similarity(rec_j["name"], parent["name"])
                            if sim_i_k >= 0.85 and sim_j_k >= 0.85:
                                tokens_i = set(P.normalize_name(rec_i["name"]).split())
                                tokens_j = set(P.normalize_name(rec_j["name"]).split())
                                tokens_k = set(P.normalize_name(parent["name"]).split())
                                if (tokens_i.issubset(tokens_k) or sim_i_k >= 0.90) and (tokens_j.issubset(tokens_k) or sim_j_k >= 0.90):
                                    alt_sims.append(min(sim_i_k, sim_j_k))

                    sim = max(alt_sims)
                pairs_scores.append(sim)

        min_score = min(pairs_scores) if pairs_scores else 1.0
        names_str = " vs ".join(f"[{r['dtype']}] {r['name']}" for r in name_records)
        if min_score >= 0.85:
            checks.append(
                CrossDocumentCheck(
                    name="Name Consistency",
                    status="MATCH",
                    summary=f"Traveller name is consistent across {len(name_records)} documents.",
                    severity="low",
                    details={"names": [r["name"] for r in name_records], "similarity": round(min_score, 2)},
                )
            )
        elif min_score >= 0.70:
            checks.append(
                CrossDocumentCheck(
                    name="Name Consistency",
                    status="PARTIAL",
                    summary=f"Minor name variation detected: {names_str}",
                    severity="medium",
                    details={"names": [r["name"] for r in name_records], "similarity": round(min_score, 2)},
                )
            )
            findings.append(
                Finding(
                    module="cross_validation",
                    severity="medium",
                    code="CROSS_DOC_NAME_PARTIAL",
                    message=f"Minor name variation across documents ({names_str}). Officer inspection recommended.",
                )
            )
        else:
            checks.append(
                CrossDocumentCheck(
                    name="Name Consistency",
                    status="MISMATCH",
                    summary=f"Name mismatch detected across documents: {names_str}",
                    severity="high",
                    details={"names": [r["name"] for r in name_records], "similarity": round(min_score, 2)},
                )
            )
            findings.append(
                Finding(
                    module="cross_validation",
                    severity="high",
                    code="CROSS_DOC_NAME_MISMATCH",
                    message=f"Traveller name does not match across documents: {names_str}.",
                )
            )
    elif len(name_records) == 1:
        checks.append(
            CrossDocumentCheck(
                name="Name Consistency",
                status="INFO",
                summary=f"Name extracted from 1 document: {name_records[0][2]}.",
                severity="info",
                details={"name": name_records[0][2]},
            )
        )

    # 2. DATE OF BIRTH CONSISTENCY
    dob_records: list[tuple[str, str, str]] = []  # (doc_label, dtype, dob_str)
    for doc in documents:
        f = doc.get("fields", {})
        doc_label = doc.get("filename") or doc.get("doc_id") or "Doc"
        dtype = doc.get("doc_type") or "Document"
        dob = f.get("date_of_birth")
        if dob:
            normalized_dob = P.normalize_date_str(dob) or dob
            dob_records.append((doc_label, str(dtype), normalized_dob))

    if len(dob_records) >= 2:
        distinct_dobs = set(d for _, _, d in dob_records)
        if len(distinct_dobs) == 1:
            common_dob = list(distinct_dobs)[0]
            checks.append(
                CrossDocumentCheck(
                    name="Date of Birth Consistency",
                    status="MATCH",
                    summary=f"Date of birth ({common_dob}) matches across all {len(dob_records)} documents.",
                    severity="low",
                    details={"dob": common_dob, "count": len(dob_records)},
                )
            )
        else:
            dob_summary = ", ".join(f"[{dtype}] {d}" for _, dtype, d in dob_records)
            checks.append(
                CrossDocumentCheck(
                    name="Date of Birth Consistency",
                    status="MISMATCH",
                    summary=f"Conflicting dates of birth detected: {dob_summary}",
                    severity="high",
                    details={"dobs": [d for _, _, d in dob_records]},
                )
            )
            findings.append(
                Finding(
                    module="cross_validation",
                    severity="high",
                    code="CROSS_DOC_DOB_MISMATCH",
                    message=f"Date of birth discrepancy detected: {dob_summary}.",
                )
            )
    elif len(dob_records) == 1:
        checks.append(
            CrossDocumentCheck(
                name="Date of Birth Consistency",
                status="INFO",
                summary=f"Date of birth found on 1 document: {dob_records[0][2]}.",
                severity="info",
                details={"dob": dob_records[0][2]},
            )
        )

    # 3. GENDER CONSISTENCY
    gender_records: list[tuple[str, str, str]] = []
    for doc in documents:
        f = doc.get("fields", {})
        g = f.get("sex")
        if g and str(g).upper() in {"M", "F", "OTHER"}:
            gender_records.append((doc.get("filename") or "Doc", str(doc.get("doc_type")), str(g).upper()))

    if len(gender_records) >= 2:
        distinct_genders = set(g for _, _, g in gender_records)
        if len(distinct_genders) == 1:
            checks.append(
                CrossDocumentCheck(
                    name="Gender Consistency",
                    status="MATCH",
                    summary=f"Gender ({list(distinct_genders)[0]}) matches across {len(gender_records)} documents.",
                    severity="low",
                    details={"gender": list(distinct_genders)[0]},
                )
            )
        else:
            g_str = ", ".join(f"[{dtype}] {g}" for _, dtype, g in gender_records)
            checks.append(
                CrossDocumentCheck(
                    name="Gender Consistency",
                    status="MISMATCH",
                    summary=f"Gender discrepancy across documents: {g_str}",
                    severity="high",
                    details={"genders": [g for _, _, g in gender_records]},
                )
            )
            findings.append(
                Finding(
                    module="cross_validation",
                    severity="high",
                    code="CROSS_DOC_GENDER_MISMATCH",
                    message=f"Conflicting gender designations found: {g_str}.",
                )
            )

    # 4. VISA-TO-PASSPORT LINKAGE
    passport_docs = docs_by_type.get("indian_passport", []) or docs_by_type.get("PASSPORT", [])
    visa_docs = docs_by_type.get("visa", []) or docs_by_type.get("VISA", [])
    if passport_docs and visa_docs:
        p_fields = passport_docs[0].get("fields", {})
        v_fields = visa_docs[0].get("fields", {})
        p_num = re.sub(r"[^A-Z0-9]", "", (p_fields.get("passport_number") or "").upper())
        v_pass = re.sub(
            r"[^A-Z0-9]", "", (v_fields.get("passport_number") or v_fields.get("travel_document_number") or "").upper()
        )

        if p_num and v_pass:
            if p_num == v_pass:
                checks.append(
                    CrossDocumentCheck(
                        name="Visa-to-Passport Linkage",
                        status="MATCH",
                        summary=f"Visa is officially linked to presented passport ({p_num}).",
                        severity="low",
                        details={"passport_number": p_num, "visa_passport_ref": v_pass},
                    )
                )
            else:
                checks.append(
                    CrossDocumentCheck(
                        name="Visa-to-Passport Linkage",
                        status="MISMATCH",
                        summary=f"Visa cites passport '{v_pass}', but presented passport is '{p_num}'.",
                        severity="high",
                        details={"passport_number": p_num, "visa_passport_ref": v_pass},
                    )
                )
                findings.append(
                    Finding(
                        module="cross_validation",
                        severity="high",
                        code="VISA_PASSPORT_MISMATCH",
                        message=f"Visa was issued for passport '{v_pass}', but presented passport is '{p_num}'.",
                    )
                )
        elif not v_pass:
            checks.append(
                CrossDocumentCheck(
                    name="Visa-to-Passport Linkage",
                    status="WARNING",
                    summary="Visa document does not show an extracted passport reference number.",
                    severity="medium",
                    details={"passport_number": p_num},
                )
            )

    # 5. TEMPORAL VALIDITY & EXPIRATION STATUS
    expired_docs = []
    active_docs = []
    now_dt = _now()
    for doc in documents:
        f = doc.get("fields", {})
        label = doc.get("filename") or str(doc.get("doc_type"))
        exp = P.parse_date(f.get("date_of_expiry") or f.get("valid_until"))
        if exp:
            if exp < now_dt:
                expired_docs.append((label, exp.strftime("%d/%m/%Y")))
            else:
                active_docs.append((label, exp.strftime("%d/%m/%Y")))

    if expired_docs:
        exp_str = ", ".join(f"{lbl} (expired {dt})" for lbl, dt in expired_docs)
        checks.append(
            CrossDocumentCheck(
                name="Document Validity Window",
                status="MISMATCH",
                summary=f"Expired document(s) detected: {exp_str}",
                severity="high",
                details={"expired": expired_docs, "active": active_docs},
            )
        )
        findings.append(
            Finding(
                module="cross_validation",
                severity="high",
                code="EXPIRED_DOCUMENT_PRESENTED",
                message=f"Traveller presented expired credential(s): {exp_str}.",
            )
        )
    elif active_docs:
        checks.append(
            CrossDocumentCheck(
                name="Document Validity Window",
                status="MATCH",
                summary=f"All {len(active_docs)} expiration-checked documents are currently valid.",
                severity="low",
                details={"active": active_docs},
            )
        )

    # 6. VISA EXPIRES AFTER PASSPORT EXPIRY CHECK
    if passport_docs and visa_docs:
        p_exp = P.parse_date(passport_docs[0].get("fields", {}).get("date_of_expiry"))
        v_exp = P.parse_date(visa_docs[0].get("fields", {}).get("valid_until"))
        if p_exp and v_exp and v_exp > p_exp:
            checks.append(
                CrossDocumentCheck(
                    name="Visa vs Passport Expiry",
                    status="WARNING",
                    summary=f"Visa valid until {v_exp.strftime('%d/%m/%Y')}, which exceeds passport expiry {p_exp.strftime('%d/%m/%Y')}.",
                    severity="medium",
                    details={"passport_expiry": p_exp.strftime("%d/%m/%Y"), "visa_expiry": v_exp.strftime("%d/%m/%Y")},
                )
            )
            findings.append(
                Finding(
                    module="cross_validation",
                    severity="medium",
                    code="VISA_EXCEEDS_PASSPORT",
                    message="Visa validity extends beyond passport expiration date.",
                )
            )

    return checks, findings

