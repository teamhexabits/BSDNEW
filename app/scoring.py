from app.schemas import Recommendation, RiskLevel, ScreeningResult


WEIGHTS = {
    "ocr": 0.15,
    "validation": 0.30,
    "tampering": 0.35,
    "face": 0.20,
}


def aggregate(result: ScreeningResult) -> ScreeningResult:
    live_missing = any(f.code == "NO_LIVE_PHOTO" for f in result.findings)
    weights = dict(WEIGHTS)
    if live_missing:
        weights["face"] = 0.0
        rest = 1.0 - WEIGHTS["face"]
        weights["ocr"] = WEIGHTS["ocr"] / rest
        weights["validation"] = WEIGHTS["validation"] / rest
        weights["tampering"] = WEIGHTS["tampering"] / rest

    score = (
        result.ocr.score * weights["ocr"]
        + result.validation.score * weights["validation"]
        + result.tampering.score * weights["tampering"]
        + result.face.score * weights["face"]
    )
    high_findings = sum(1 for f in result.findings if f.severity == "high")
    if high_findings >= 2:
        score = min(score, 48)
    elif high_findings == 1:
        score = min(score, 68)

    if score >= 80:
        risk, rec = RiskLevel.LOW, Recommendation.ALLOW
    elif score >= 55:
        risk, rec = RiskLevel.MEDIUM, Recommendation.SECONDARY_INSPECTION
    else:
        risk, rec = RiskLevel.HIGH, Recommendation.REFER_FRAUD_CELL

    result.authenticity_score = round(score, 1)
    result.risk_level = risk
    result.recommendation = rec
    return result


def aggregate_batch(batch: "BatchScreeningResult") -> "BatchScreeningResult":
    from app.schemas import Recommendation, RiskLevel

    if not batch.documents:
        batch.overall_authenticity_score = 0.0
        batch.risk_level = RiskLevel.HIGH
        batch.recommendation = Recommendation.REFER_FRAUD_CELL
        batch.summary = "No documents submitted."
        return batch

    # Average individual doc scores
    doc_scores = [d.authenticity_score for d in batch.documents]
    base_score = sum(doc_scores) / len(doc_scores)

    # Check cross-checks & adverse findings
    cross_high = [c for c in batch.cross_checks if c.severity == "high" and c.status == "MISMATCH"]
    cross_medium = [c for c in batch.cross_checks if c.severity == "medium"]

    total_high = sum(1 for f in batch.total_findings if f.severity == "high")
    total_medium = sum(1 for f in batch.total_findings if f.severity == "medium")

    score = base_score
    if cross_high:
        score = min(score, 42.0)
    elif total_high >= 3:
        score = min(score, 45.0)
    elif total_high >= 1:
        score = min(score, 62.0)
    elif cross_medium or total_medium >= 2:
        score = min(score, 74.0)

    score = max(0.0, min(100.0, score))

    if score >= 80:
        risk, rec = RiskLevel.LOW, Recommendation.ALLOW
        summary = f"Low risk. Verified {len(batch.documents)} document(s) with high identity consistency."
    elif score >= 55:
        risk, rec = RiskLevel.MEDIUM, Recommendation.SECONDARY_INSPECTION
        summary = f"Medium risk. Secondary officer inspection recommended for {len(batch.documents)} document(s)."
    else:
        risk, rec = RiskLevel.HIGH, Recommendation.REFER_FRAUD_CELL
        summary = f"High risk. Critical discrepancies or tamper/validation anomalies detected across documents."

    batch.overall_authenticity_score = round(score, 1)
    batch.risk_level = risk
    batch.recommendation = rec
    batch.summary = summary
    return batch

