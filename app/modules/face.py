"""One-to-one face verification: document photo vs webcam capture (InsightFace buffalo_l)."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from insightface.app import FaceAnalysis

from app.schemas import Finding, ModuleResult

logger = logging.getLogger(__name__)

# Starting cosine-similarity threshold for buffalo_l embeddings (L2-normalized).
# This is NOT a percentage: 0.40 is a cosine score, not "40%".
# Calibrate with genuine and impostor pairs before any operational use.
FACE_MATCH_THRESHOLD = 0.40

# Faces with a lower detector score are treated as unusable.
_MIN_DET_SCORE = 0.30
# On documents, pick the largest face only if it clearly dominates the next one.
_DOCUMENT_DOMINANT_AREA_RATIO = 1.5

_face_app: FaceAnalysis | None = None


def _init_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_face_app() -> FaceAnalysis:
    """Return the process-wide buffalo_l analyzer (loaded once, not per request)."""
    global _face_app
    if _face_app is None:
        logger.info("Loading InsightFace buffalo_l (CPU)…")
        _face_app = _init_face_app()
        logger.info("InsightFace buffalo_l ready.")
    return _face_app


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode upload bytes to an OpenCV BGR color image (no grayscale conversion)."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _crop_norm_bgr(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(width * x0)))
    top = max(0, min(height - 1, int(height * y0)))
    right = max(left + 1, min(width, int(width * x1)))
    bottom = max(top + 1, min(height, int(height * y1)))
    return image[top:bottom, left:right]


# Indian passport information page: holder photograph is the left panel.
_PASSPORT_PORTRAIT_BOX = (0.00, 0.05, 0.36, 0.72)


def _document_image_for_faces(document_bytes: bytes, document_type: str | None) -> np.ndarray:
    image = bytes_to_bgr(document_bytes)
    if document_type in {"indian_passport", "PASSPORT"}:
        portrait = _crop_norm_bgr(image, _PASSPORT_PORTRAIT_BOX)
        print("Face source: passport left-panel photograph")
        return portrait
    return image


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, float(x2 - x1) * float(y2 - y1))


def _usable_faces(faces: list) -> list:
    usable = []
    for face in faces:
        if float(getattr(face, "det_score", 0.0)) < _MIN_DET_SCORE:
            continue
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raw = getattr(face, "embedding", None)
            if raw is None:
                continue
        usable.append(face)
    return usable


def _normalized_embedding(face) -> np.ndarray:
    embedding = getattr(face, "normed_embedding", None)
    if embedding is not None:
        return np.asarray(embedding, dtype=np.float64)
    raw = np.asarray(face.embedding, dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if norm == 0.0:
        raise ValueError("Face embedding has zero norm.")
    return raw / norm


def _select_document_face(faces: list, *, pick_largest: bool = False) -> tuple[object | None, str | None]:
    """Largest / highest-quality face on a document, or error if the choice is ambiguous."""
    if not faces:
        return None, "No usable face detected in the document photograph."
    if len(faces) == 1 or pick_largest:
        return max(faces, key=_bbox_area), None
    ranked = sorted(faces, key=_bbox_area, reverse=True)
    largest, second = ranked[0], ranked[1]
    if _bbox_area(second) <= 0:
        return largest, None
    if _bbox_area(largest) / _bbox_area(second) < _DOCUMENT_DOMINANT_AREA_RATIO:
        return None, "Multiple faces detected on the document; result would be ambiguous."
    return largest, None


def _select_live_face(faces: list) -> tuple[object | None, str | None]:
    if not faces:
        return None, "No face detected in live capture"
    if len(faces) > 1:
        return None, "Multiple faces detected"
    return faces[0], None


@dataclass(frozen=True)
class FaceRead:
    embedding: np.ndarray | None
    count: int
    error: str | None


def get_face_embedding(image: np.ndarray, *, source: str, pick_largest: bool = False) -> FaceRead:
    """Detect faces on a color BGR image and return the selected normalized embedding."""
    faces = get_face_app().get(image)
    usable = _usable_faces(list(faces))
    if source == "live":
        chosen, error = _select_live_face(usable)
    else:
        chosen, error = _select_document_face(usable, pick_largest=pick_largest)
    if error or chosen is None:
        return FaceRead(embedding=None, count=len(usable), error=error)
    return FaceRead(embedding=_normalized_embedding(chosen), count=len(usable), error=None)


def _confidence_from_similarity(similarity: float) -> float:
    """How far the cosine score sits from the threshold, mapped to 0–100 (not a percentage match)."""
    gap = abs(similarity - FACE_MATCH_THRESHOLD)
    return float(min(100.0, max(55.0, 55.0 + gap * 160.0)))


def _log_verify(
    document_count: int,
    live_count: int,
    similarity: float | None,
    matched: bool,
    failed: bool,
) -> None:
    result = "FAILED" if failed else ("MATCH" if matched else "MISMATCH")
    sim_text = "n/a" if similarity is None else f"{similarity:.4f}"
    lines = (
        f"Document faces: {document_count}",
        f"Live faces: {live_count}",
        f"Similarity: {sim_text}",
        f"Threshold: {FACE_MATCH_THRESHOLD:.2f}",
        f"Result: {result}",
    )
    message = "\n".join(lines)
    print(message)
    logger.info(message)


def verify_faces(
    document_bytes: bytes,
    live_bytes: bytes,
    document_type: str | None = None,
) -> dict:
    """Compare the document photograph to a webcam capture. One-to-one only (no gallery search)."""
    document_image = _document_image_for_faces(document_bytes, document_type)
    live_image = bytes_to_bgr(live_bytes)

    document = get_face_embedding(
        document_image,
        source="document",
        pick_largest=document_type in {"indian_passport", "PASSPORT"},
    )
    live = get_face_embedding(live_image, source="live")

    if document.error:
        _log_verify(document.count, live.count, None, False, True)
        return {
            "matched": False,
            "similarity": None,
            "threshold": FACE_MATCH_THRESHOLD,
            "confidence": 100.0,
            "message": "FACE VERIFICATION FAILED",
            "reason": document.error,
        }

    if live.error:
        _log_verify(document.count, live.count, None, False, True)
        return {
            "matched": False,
            "similarity": None,
            "threshold": FACE_MATCH_THRESHOLD,
            "confidence": 100.0,
            "message": "FACE VERIFICATION FAILED",
            "reason": live.error,
        }

    similarity = float(np.dot(document.embedding, live.embedding))
    matched = similarity >= FACE_MATCH_THRESHOLD
    _log_verify(document.count, live.count, similarity, matched, False)
    return {
        "matched": matched,
        "similarity": round(similarity, 4),
        "threshold": FACE_MATCH_THRESHOLD,
        "confidence": round(_confidence_from_similarity(similarity), 1),
        "message": "FACE MATCH" if matched else "FACE MISMATCH",
        "reason": "match" if matched else "mismatch",
    }


def run_face(
    document_bytes: bytes,
    live_bytes: bytes | None,
    document_type: str | None = None,
) -> tuple[ModuleResult, list[Finding]]:
    findings: list[Finding] = []

    if not live_bytes:
        findings.append(
            Finding(
                module="face",
                severity="medium",
                code="NO_LIVE_PHOTO",
                message="No live capture provided — face verification skipped.",
            )
        )
        return (
            ModuleResult(
                name="face",
                score=50.0,
                confidence=20.0,
                summary="Live photograph missing; face verification skipped.",
                details={
                    "matched": False,
                    "reason": "missing_live_photo",
                    "method": "InsightFace buffalo_l",
                    "threshold": FACE_MATCH_THRESHOLD,
                },
            ),
            findings,
        )

    payload = verify_faces(document_bytes, live_bytes, document_type)
    reason = payload["reason"]
    similarity = payload["similarity"]
    matched = bool(payload["matched"])
    confidence = float(payload["confidence"])

    if similarity is None:
        code_map = {
            "No usable face detected in the document photograph.": "DOCUMENT_FACE_NOT_FOUND",
            "Multiple faces detected on the document; result would be ambiguous.": "DOCUMENT_FACES_AMBIGUOUS",
            "No face detected in live capture": "LIVE_FACE_NOT_FOUND",
            "Multiple faces detected": "LIVE_MULTIPLE_FACES",
        }
        code = code_map.get(reason or "", "FACE_VERIFY_FAILED")
        findings.append(
            Finding(
                module="face",
                severity="high",
                code=code,
                message=reason or "Face verification failed.",
            )
        )
        return (
            ModuleResult(
                name="face",
                score=0.0,
                confidence=confidence,
                summary=reason or "Face verification failed.",
                details={
                    "matched": False,
                    "similarity": None,
                    "threshold": FACE_MATCH_THRESHOLD,
                    "reason": reason,
                    "method": "InsightFace buffalo_l cosine similarity",
                },
            ),
            findings,
        )

    portrait = document_type in {"indian_passport", "PASSPORT"}
    if matched:
        score = 100.0
        summary = (
            "Passport photograph and live capture match."
            if portrait
            else "Document photo and live capture match."
        )
    else:
        score = 10.0
        findings.append(
            Finding(
                module="face",
                severity="high",
                code="FACE_MISMATCH",
                message=(
                    "Live capture does not match the passport photograph."
                    if portrait
                    else "Live capture does not match the document photograph."
                ),
            )
        )
        summary = (
            "Passport photograph and live capture appear to be different people."
            if portrait
            else "Document photo and live capture appear to be different people."
        )

    return (
        ModuleResult(
            name="face",
            score=score,
            confidence=confidence,
            summary=summary,
            details={
                "matched": matched,
                "similarity": similarity,
                "threshold": FACE_MATCH_THRESHOLD,
                "reason": reason,
                "method": "InsightFace buffalo_l cosine similarity",
            },
        ),
        findings,
    )


def extract_document_face_embedding(
    document_bytes: bytes, document_type: str | None = None
) -> tuple[np.ndarray | None, str | None]:
    """Extract normalized face embedding from document image bytes."""
    try:
        doc_img = _document_image_for_faces(document_bytes, document_type)
        read = get_face_embedding(
            doc_img,
            source="document",
            pick_largest=document_type in {"indian_passport", "PASSPORT"},
        )
        return read.embedding, read.error
    except Exception as exc:
        return None, str(exc)


def verify_batch_faces(
    documents: list[dict[str, Any]],
    live_bytes: bytes | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[Finding]]:
    from app.schemas import CrossDocFaceMatch

    cross_matches: list[CrossDocFaceMatch] = []
    live_matches: list[dict[str, Any]] = []
    findings: list[Finding] = []

    # Extract embeddings for all documents
    embeddings: list[dict[str, Any]] = []
    for doc in documents:
        b = doc.get("image_bytes")
        if not b:
            continue
        dtype = str(doc.get("doc_type") or doc.get("document_type") or "")
        emb, err = extract_document_face_embedding(b, dtype)
        if emb is not None:
            embeddings.append({
                "doc_id": doc.get("doc_id") or doc.get("filename") or "doc",
                "filename": doc.get("filename") or "document",
                "doc_type": dtype,
                "embedding": emb,
            })

    # Pairwise cross-document face comparison
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d1, d2 = embeddings[i], embeddings[j]
            sim = float(np.dot(d1["embedding"], d2["embedding"]))
            matched = sim >= FACE_MATCH_THRESHOLD
            cross_matches.append(
                CrossDocFaceMatch(
                    doc1_id=d1["filename"],
                    doc1_type=d1["doc_type"],
                    doc2_id=d2["filename"],
                    doc2_type=d2["doc_type"],
                    similarity=round(sim, 4),
                    matched=matched,
                    threshold=FACE_MATCH_THRESHOLD,
                )
            )
            if not matched:
                findings.append(
                    Finding(
                        module="face",
                        severity="high",
                        code="CROSS_DOC_FACE_MISMATCH",
                        message=(
                            f"Face in {d1['filename']} ({d1['doc_type']}) does not match "
                            f"face in {d2['filename']} ({d2['doc_type']}) [similarity: {sim:.2f}]."
                        ),
                    )
                )

    # Live face comparison
    if live_bytes:
        try:
            live_image = bytes_to_bgr(live_bytes)
            live = get_face_embedding(live_image, source="live")
            if live.embedding is not None:
                for d in embeddings:
                    sim = float(np.dot(d["embedding"], live.embedding))
                    matched = sim >= FACE_MATCH_THRESHOLD
                    live_matches.append({
                        "doc_id": d["doc_id"],
                        "filename": d["filename"],
                        "doc_type": d["doc_type"],
                        "similarity": round(sim, 4),
                        "matched": matched,
                        "threshold": FACE_MATCH_THRESHOLD,
                    })
                    if not matched:
                        findings.append(
                            Finding(
                                module="face",
                                severity="high",
                                code="LIVE_DOC_FACE_MISMATCH",
                                message=(
                                    f"Live webcam photo does not match photograph in {d['filename']} "
                                    f"({d['doc_type']}) [similarity: {sim:.2f}]."
                                ),
                            )
                        )
            else:
                findings.append(
                    Finding(
                        module="face",
                        severity="medium",
                        code="LIVE_FACE_NOT_FOUND",
                        message=live.error or "Could not extract face from live capture.",
                    )
                )
        except Exception as exc:
            findings.append(
                Finding(
                    module="face",
                    severity="medium",
                    code="LIVE_FACE_ERROR",
                    message=f"Error analyzing live face: {exc}",
                )
            )

    return cross_matches, live_matches, findings

