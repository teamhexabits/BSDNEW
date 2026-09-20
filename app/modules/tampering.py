"""Image-forensics heuristics for reprint, splice, and clone-style tampering."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageStat

from app.schemas import Finding, ModuleResult


def _pil(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _ela_score(image: Image.Image) -> tuple[float, float]:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela = ImageChops.difference(image, resaved)
    extrema = ela.getextrema()
    max_diff = max(channel[1] for channel in extrema)
    scale = 255.0 / max_diff if max_diff else 1.0
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    stat = ImageStat.Stat(ela)
    mean_energy = float(sum(stat.mean) / len(stat.mean))
    # High local energy after resave is typical of splices / heavy edits.
    risk = min(100.0, mean_energy * 1.8)
    authenticity = max(0.0, 100.0 - risk)
    return authenticity, mean_energy


def _to_gray(arr: np.ndarray) -> np.ndarray:
    return np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def _noise_inconsistency(arr: np.ndarray) -> float:
    gray = _to_gray(arr)
    blur = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=1)))
    noise = np.abs(gray.astype(np.int16) - blur.astype(np.int16)).astype(np.float32)
    h, w = noise.shape
    tiles = []
    step_y, step_x = max(h // 4, 8), max(w // 4, 8)
    for y in range(0, h - step_y + 1, step_y):
        for x in range(0, w - step_x + 1, step_x):
            tiles.append(float(noise[y : y + step_y, x : x + step_x].std()))
    if not tiles:
        return 70.0
    spread = float(np.std(tiles) / (np.mean(tiles) + 1e-6))
    return max(0.0, 100.0 - min(100.0, spread * 80))


def _blur_reprint(arr: np.ndarray) -> float:
    gray = _to_gray(arr).astype(np.float64)
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    variance = float(lap.var())
    # Screen-recapture / photocopy tends to collapse high-frequency detail.
    if variance < 40:
        return 35.0
    if variance < 80:
        return 60.0
    return min(100.0, 55.0 + variance / 20.0)


def _clone_blocks(arr: np.ndarray) -> float:
    gray = _to_gray(arr)
    small = np.array(Image.fromarray(gray).resize((128, 128), Image.Resampling.BILINEAR))
    block = 16
    hashes: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for y in range(0, 128 - block, block // 2):
        for x in range(0, 128 - block, block // 2):
            patch = small[y : y + block, x : x + block]
            key = tuple((patch[::4, ::4] // 16).flatten().tolist())
            hashes.setdefault(key, []).append((x, y))
    clones = 0
    for positions in hashes.values():
        if len(positions) < 2:
            continue
        for i, (x1, y1) in enumerate(positions):
            for x2, y2 in positions[i + 1 :]:
                dist = abs(x1 - x2) + abs(y1 - y2)
                if dist >= block:
                    clones += 1
    if clones >= 6:
        return 40.0
    if clones >= 3:
        return 65.0
    return 90.0


def _exif_flags(data: bytes) -> tuple[float, list[str]]:
    notes: list[str] = []
    try:
        image = Image.open(io.BytesIO(data))
        exif = image.getexif()
        software = str(exif.get(305, "")).lower() if exif else ""
        if any(tag in software for tag in ("photoshop", "gimp", "snapseed", "picsart")):
            notes.append(f"Editor software in EXIF: {software}")
            return 45.0, notes
        if not exif:
            notes.append("No EXIF metadata — common for scans and composites.")
            return 70.0, notes
        return 88.0, notes
    except Exception:
        return 70.0, notes


def run_tampering(image_bytes: bytes) -> tuple[ModuleResult, list[Finding]]:
    image = _pil(image_bytes)
    arr = np.array(image)
    findings: list[Finding] = []

    ela_auth, ela_energy = _ela_score(image)
    noise_auth = _noise_inconsistency(arr)
    blur_auth = _blur_reprint(arr)
    clone_auth = _clone_blocks(arr)
    exif_auth, exif_notes = _exif_flags(image_bytes)

    scores = {
        "error_level_analysis": round(ela_auth, 1),
        "noise_consistency": round(noise_auth, 1),
        "print_recapture": round(blur_auth, 1),
        "clone_blocks": round(clone_auth, 1),
        "metadata": round(exif_auth, 1),
    }
    weights = {
        "error_level_analysis": 0.30,
        "noise_consistency": 0.20,
        "print_recapture": 0.20,
        "clone_blocks": 0.20,
        "metadata": 0.10,
    }
    authenticity = sum(scores[k] * weights[k] for k in scores)

    if ela_auth < 55:
        findings.append(
            Finding(
                module="tampering",
                severity="high",
                code="ELA_ANOMALY",
                message="Error-level analysis shows uneven JPEG residuals (possible splice or stamp edit).",
            )
        )
    if blur_auth < 50:
        findings.append(
            Finding(
                module="tampering",
                severity="high",
                code="RECAPTURE",
                message="Low micro-texture suggests a photocopy or screen recapture rather than an original scan.",
            )
        )
    if clone_auth < 70:
        findings.append(
            Finding(
                module="tampering",
                severity="medium",
                code="CLONE_REGION",
                message="Repeated image blocks detected — check photo, stamp, or signature regions.",
            )
        )
    for note in exif_notes:
        findings.append(
            Finding(module="tampering", severity="low", code="METADATA", message=note)
        )

    result = ModuleResult(
        name="tampering",
        score=round(authenticity, 1),
        confidence=75.0,
        summary="Forensic pass across ELA, noise, recapture, clone, and metadata signals.",
        details={"signals": scores, "ela_mean_energy": round(ela_energy, 2)},
    )
    return result, findings
