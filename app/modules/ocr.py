"""OCR extraction for passport, visa, Aadhaar, and driving licence."""

from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.documents import patterns as P
from app.schemas import DocumentType, Finding, ModuleResult

_MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
_PASSPORT_LABELED = re.compile(
    r"(?:PASSPORT\s*(?:NO\.?|NUMBER|#)|PASSPORT NO\.?)\s*[:.\-]?\s*([A-Z]{1,2}\s*[0-9]{6,7})"
)
_VISA_LABELED = re.compile(
    r"(?:VISA\s*(?:NO\.?|NUMBER|#)?)\s*[:.\-]?\s*([A-Z]{1,3}\s*[0-9]{6,8}|[A-Z0-9]{7,12})"
)


def _to_pil(data: bytes) -> Image.Image:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:
        pass
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _windows_tesseract() -> bool:
    return os.name == "nt"


def _usable_binary(path: Path) -> bool:
    if not path.is_file():
        return False
    if _windows_tesseract():
        return path.suffix.lower() == ".exe"
    return True


def _tesseract_candidates() -> list[Path]:
    prefix = _project_root() / ".tools" / "tesseract"
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    program_files_x86 = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    local_app = Path(os.environ.get("LOCALAPPDATA", ""))
    return [
        prefix / "tesseract.exe",
        prefix / "bin" / "tesseract.exe",
        prefix / "bin" / "tesseract",
        program_files / "Tesseract-OCR" / "tesseract.exe",
        program_files_x86 / "Tesseract-OCR" / "tesseract.exe",
        local_app / "Programs" / "Tesseract-OCR" / "tesseract.exe",
        Path("/opt/homebrew/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ]


def _tessdata_candidates(cmd: Path | None = None) -> list[Path]:
    prefix = _project_root() / ".tools" / "tesseract"
    found: list[Path] = [prefix / "tessdata"]
    if cmd is not None:
        found.extend(
            [
                cmd.parent / "tessdata",
                cmd.parent.parent / "share" / "tessdata",
            ]
        )
    found.append(prefix / "share" / "tessdata")
    return found


def configure_ocr_engine() -> str | None:
    """Point pytesseract at a Windows or local Tesseract install."""
    import pytesseract

    cmd: Path | None = None
    which = shutil.which("tesseract")
    if which and _usable_binary(Path(which)):
        cmd = Path(which)
    else:
        for candidate in _tesseract_candidates():
            if _usable_binary(candidate):
                cmd = candidate
                break

    if cmd is None:
        return None

    tessdata = next((path for path in _tessdata_candidates(cmd) if path.is_dir()), None)
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = str(tessdata)

    bin_dir = str(cmd.parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts]) if path_parts != [""] else bin_dir

    pytesseract.pytesseract.tesseract_cmd = str(cmd)
    return str(cmd)


def _tessdata_dir() -> Path | None:
    cmd = None
    try:
        import pytesseract

        configured = pytesseract.pytesseract.tesseract_cmd
        if configured:
            cmd = Path(configured)
    except Exception:
        pass
    for path in _tessdata_candidates(cmd):
        if path.is_dir():
            return path
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix and Path(prefix).is_dir():
        return Path(prefix)
    return None


def _tesseract_config(extra: str) -> str:
    return extra


def _configure_tesseract() -> None:
    configure_ocr_engine()


def _has_traineddata(lang: str) -> bool:
    tessdata = _tessdata_dir()
    if tessdata and (tessdata / f"{lang}.traineddata").is_file():
        return True
    try:
        import pytesseract

        return lang in pytesseract.get_languages(config=_tesseract_config(""))
    except Exception:
        return lang == "eng"


def _ocr_languages(document_type: DocumentType | None) -> str:
    if document_type == DocumentType.NATIONAL_ID and _has_traineddata("hin"):
        return "eng+hin"
    return "eng"


def _fit(image: Image.Image, max_side: int = 1800, min_side: int = 1200) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    shortest = min(w, h)
    if longest > max_side:
        scale = max_side / longest
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    if shortest < min_side:
        scale = min_side / shortest
        return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image


def _variants(image: Image.Image) -> list[Image.Image]:
    gray = ImageOps.autocontrast(_fit(image.convert("L")))
    arr = np.array(gray)
    variants = [gray, gray.filter(ImageFilter.SHARPEN)]
    if float(arr.mean()) < 70:
        variants.append(ImageOps.invert(gray))
    return variants


def _read_text(
    image: Image.Image,
    document_type: DocumentType | None = None,
) -> tuple[str, float]:
    """Best-effort OCR using the bundled Tesseract engine."""
    try:
        import pytesseract

        cmd = configure_ocr_engine()
        if not cmd:
            print("OCR: Tesseract binary not found")
            return "", 0.0
        lang = _ocr_languages(document_type)
        chunks: list[str] = []
        confs: list[int] = []
        last_error = None
        configs = (
            _tesseract_config("--oem 3 --psm 6"),
            _tesseract_config("--oem 3 --psm 3"),
            _tesseract_config("--oem 3 --psm 11"),
        )
        for variant in _variants(image):
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(variant, lang=lang, config=cfg)
                except Exception as exc:
                    last_error = exc
                    continue
                if text and text.strip():
                    chunks.append(text)
        if not chunks:
            if last_error:
                print(f"OCR failed: {last_error}")
            else:
                print("OCR: Tesseract returned no text")
            return "", 0.0
        seen: set[str] = set()
        merged: list[str] = []
        for chunk in chunks:
            for line in chunk.splitlines():
                key = re.sub(r"\s+", "", line.upper())
                if key and key not in seen:
                    seen.add(key)
                    merged.append(line)
        try:
            data = pytesseract.image_to_data(
                _variants(image)[0],
                lang=lang,
                output_type=pytesseract.Output.DICT,
                config=_tesseract_config("--oem 3 --psm 6"),
            )
            confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
        except Exception:
            confs = []
        confidence = float(sum(confs) / len(confs)) if confs else 55.0
        merged_text = "\n".join(merged)
        print(f"OCR chars: {len(merged_text)}  confidence: {confidence:.1f}  lang: {lang}")
        return merged_text, confidence
    except Exception as exc:
        print(f"OCR exception: {exc}")
        return "", 0.0


_PASSPORT_VISUAL_BOX = (0.28, 0.04, 0.99, 0.82)
_PASSPORT_DATES_BOX = (0.32, 0.54, 0.99, 0.72)
_PASSPORT_MRZ_BOX = (0.00, 0.68, 1.00, 1.00)
_PASSPORT_OUTPUT_KEYS = (
    "document_type",
    "issuing_country",
    "passport_number",
    "mrz_number",
    "surname",
    "given_names",
    "full_name",
    "nationality",
    "date_of_birth",
    "sex",
    "place_of_birth",
    "place_of_issue",
    "date_of_issue",
    "date_of_expiry",
    "mrz_line1",
    "mrz_line2",
    "mrz_present",
)
_PASSPORT_JUNK = {
    "REPUBLIC", "INDIA", "BHARAT", "GANARAJYA", "PASSPORT", "TYPE", "CODE",
    "NATIONALITY", "SIGNATURE", "GOVERNMENT",
}


def _crop_norm(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = box
    left = max(0, min(w - 1, int(w * x0)))
    top = max(0, min(h - 1, int(h * y0)))
    right = max(left + 1, min(w, int(w * x1)))
    bottom = max(top + 1, min(h, int(h * y1)))
    return image.crop((left, top, right, bottom))


def _passport_regions(image: Image.Image) -> dict[str, Image.Image]:
    """Crop Indian passport visual zone, dates block, and MRZ using resolution-independent boxes."""
    normalized = _fit(image.convert("RGB"))
    return {
        "visual": _crop_norm(normalized, _PASSPORT_VISUAL_BOX),
        "dates": _crop_norm(normalized, _PASSPORT_DATES_BOX),
        "mrz": _crop_norm(normalized, _PASSPORT_MRZ_BOX),
    }


def _preprocess_visual(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"))
    return gray.filter(ImageFilter.SHARPEN)


def _preprocess_mrz(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(_fit(image.convert("L"), max_side=2400, min_side=1000))
    return gray.filter(ImageFilter.SHARPEN)


def _mrz_variants(image: Image.Image) -> list[Image.Image]:
    sharp = _preprocess_mrz(image)
    variants = [sharp, ImageOps.autocontrast(sharp)]
    arr = np.array(sharp)
    if float(arr.mean()) < 90:
        variants.append(ImageOps.invert(sharp))
    variants.append(sharp.point(lambda p: 255 if p > 140 else 0))
    return variants


def _ocr_region(
    image: Image.Image | list[Image.Image],
    lang: str,
    configs: tuple[str, ...],
) -> tuple[str, float]:
    import pytesseract

    if not configure_ocr_engine():
        return "", 0.0
    images = image if isinstance(image, list) else [image]
    chunks: list[str] = []
    confs: list[int] = []
    for variant in images:
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(variant, lang=lang, config=cfg)
            except Exception as exc:
                print(f"OCR region failed: {exc}")
                continue
            if text and text.strip():
                chunks.append(text)
            try:
                data = pytesseract.image_to_data(
                    variant, lang=lang, output_type=pytesseract.Output.DICT, config=cfg
                )
                confs.extend(int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0)
            except Exception:
                pass
    seen: set[str] = set()
    merged: list[str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            key = re.sub(r"\s+", "", line.upper())
            if key and key not in seen:
                seen.add(key)
                merged.append(line)
    confidence = float(sum(confs) / len(confs)) if confs else (55.0 if merged else 0.0)
    return "\n".join(merged), confidence


def _read_passport_regions(image: Image.Image) -> tuple[str, str, float]:
    """OCR the visual data block, dedicated dates block, and MRZ."""
    regions = _passport_regions(image)
    visual_img = _preprocess_visual(regions["visual"])
    visual_text, visual_conf = _ocr_region(
        visual_img,
        "eng",
        (
            _tesseract_config("--oem 3 --psm 6"),
            _tesseract_config("--oem 3 --psm 11"),
        ),
    )
    if "dates" in regions:
        dates_img = _preprocess_visual(regions["dates"])
        dates_text, dates_conf = _ocr_region(
            dates_img,
            "eng",
            (
                _tesseract_config("--oem 3 --psm 6"),
            ),
        )
        if dates_text and dates_text.strip():
            visual_text = f"{visual_text}\n{dates_text}"
            visual_conf = max(visual_conf, dates_conf)
    mrz_text, mrz_conf = _ocr_region(
        _mrz_variants(regions["mrz"]),
        "eng",
        (
            _tesseract_config(
                f"--oem 3 --psm 6 -c tessedit_char_whitelist={_MRZ_WHITELIST}"
            ),
            _tesseract_config(
                f"--oem 3 --psm 4 -c tessedit_char_whitelist={_MRZ_WHITELIST}"
            ),
            _tesseract_config(
                f"--oem 3 --psm 11 -c tessedit_char_whitelist={_MRZ_WHITELIST}"
            ),
            _tesseract_config(
                f"--oem 3 --psm 3 -c tessedit_char_whitelist={_MRZ_WHITELIST}"
            ),
        ),
    )
    confidence = max(visual_conf, mrz_conf)
    print(
        f"Passport OCR visual chars: {len(visual_text)}  "
        f"MRZ chars: {len(_clean_mrz_line(mrz_text))}  conf: {confidence:.1f}"
    )
    return visual_text, mrz_text, confidence


def _read_visa_regions(image: Image.Image) -> tuple[str, str, float]:
    """OCR the visa document visual zones (header, identity, dates/doc, footer) and MRV line if present."""
    w, h = image.size
    chunks: list[str] = []
    confs: list[float] = []

    # 1. Identity / Name strip (y: 0.18 to 0.32, x: 0.35 to 0.98)
    name_crop = image.crop((int(w * 0.35), int(h * 0.18), int(w * 0.98), int(h * 0.32)))
    name_up = ImageOps.autocontrast(
        name_crop.resize((name_crop.width * 2, name_crop.height * 2), Image.Resampling.LANCZOS).convert("L")
    )
    t_name, c_name = _ocr_region(
        [name_up], "eng", (_tesseract_config("--oem 3 --psm 6"), _tesseract_config("--oem 3 --psm 11"))
    )
    if t_name.strip():
        chunks.append(t_name)
        confs.append(c_name)

    # 2. DOB, Sex, Nationality strip (y: 0.28 to 0.42, x: 0.35 to 0.98)
    dob_crop = image.crop((int(w * 0.35), int(h * 0.28), int(w * 0.98), int(h * 0.42)))
    dob_up = ImageOps.autocontrast(
        dob_crop.resize((dob_crop.width * 2, dob_crop.height * 2), Image.Resampling.LANCZOS).convert("L")
    )
    t_dob, c_dob = _ocr_region(
        [dob_up], "eng", (_tesseract_config("--oem 3 --psm 4"), _tesseract_config("--oem 3 --psm 11"))
    )
    if t_dob.strip():
        chunks.append(t_dob)
        confs.append(c_dob)

    # 3. Dates & Travel Doc strip (y: 0.35 to 0.62, x: 0.35 to 0.98)
    dates_crop = image.crop((int(w * 0.35), int(h * 0.35), int(w * 0.98), int(h * 0.62)))
    dates_up = ImageOps.autocontrast(
        dates_crop.resize((dates_crop.width * 3, dates_crop.height * 3), Image.Resampling.LANCZOS).convert("L")
    )
    t_dates, c_dates = _ocr_region(
        [dates_up], "eng", (_tesseract_config("--oem 3 --psm 11"), _tesseract_config("--oem 3 --psm 4"))
    )
    if t_dates.strip():
        chunks.append(t_dates)
        confs.append(c_dates)

    # 4. Header Zone (y: 0.0 to 0.22, x: 0.35 to 0.99)
    header_crop = image.crop((int(w * 0.35), int(h * 0.0), int(w * 0.99), int(h * 0.22)))
    header_up = ImageOps.autocontrast(
        header_crop.resize((header_crop.width * 2, header_crop.height * 2), Image.Resampling.LANCZOS).convert("L")
    )
    t_hdr, c_hdr = _ocr_region(
        [header_up], "eng", (_tesseract_config("--oem 3 --psm 6"), _tesseract_config("--oem 3 --psm 11"))
    )
    if t_hdr.strip():
        chunks.append(t_hdr)
        confs.append(c_hdr)

    # 5. Footer Zone (y: 0.62 to 0.99, x: 0.35 to 0.99)
    footer_crop = image.crop((int(w * 0.35), int(h * 0.62), int(w * 0.99), int(h * 0.99)))
    footer_up = ImageOps.autocontrast(
        footer_crop.resize((footer_crop.width * 2, footer_crop.height * 2), Image.Resampling.LANCZOS).convert("L")
    )
    t_ftr, c_ftr = _ocr_region(
        [footer_up], "eng", (_tesseract_config("--oem 3 --psm 6"), _tesseract_config("--oem 3 --psm 11"))
    )
    if t_ftr.strip():
        chunks.append(t_ftr)
        confs.append(c_ftr)

    # 6. Full document context
    full_text, full_conf = _read_text(image, DocumentType.VISA)
    if full_text.strip():
        chunks.append(full_text)
        confs.append(full_conf)

    # 7. MRZ Zone (check bottom 20% for V< Machine-Readable Visa lines)
    mrz_crop = image.crop((int(w * 0.01), int(h * 0.78), int(w * 0.99), int(h * 0.99)))
    t_mrz, c_mrz = _ocr_region([mrz_crop], "eng", (_tesseract_config("--oem 3 --psm 6"),))
    mrz_text = t_mrz if "V<" in _clean_mrz_line(t_mrz) else ""

    visual_text = "\n".join(chunks)
    confidence = float(sum(confs) / len(confs)) if confs else 50.0
    print(
        f"Visa OCR visual chars: {len(visual_text)}  "
        f"MRZ chars: {len(_clean_mrz_line(mrz_text))}  conf: {confidence:.1f}"
    )
    return visual_text, mrz_text, confidence


_FIELD_WORDS = {
    "SURNAME", "GIVEN", "NAME", "NAMES", "NATIONALITY", "SEX", "GENDER",
    "DATE", "BIRTH", "PLACE", "ISSUE", "EXPIRY", "PASSPORT", "COUNTRY",
    "CODE", "TYPE", "DOB", "DOI", "ADDRESS", "BLOOD",
}
_TWO_DATES = re.compile(
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})\s+(\d{1,2}[./-]\d{1,2}[./-]\d{4})"
)
_LOOSE_DMY = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\b")
_EXPIRY_LABELS = (
    r"DATE OF EXPIRY|DATE OF EXPIRATION|DATE OF EXPIRE|DATE OF EXPLRY|"
    r"EXPIRY DATE|\bEXPIRY\b|VALID TILL|VALID UNTIL|VALID TO"
)


def _first(regex: re.Pattern[str], text: str) -> str | None:
    match = regex.search(text.upper())
    if not match:
        return None
    return re.sub(r"\s+", "", match.group(1))


def _strip_bilingual(value: str) -> str:
    value = re.sub(r"[\u0900-\u097F]+", " ", value)
    value = re.sub(r"(?<=[A-Za-z])\s*/\s*(?=[A-Za-z\u0900-\u097F])", " ", value)
    value = re.sub(r"^\s*/\s*|\s*/\s*$", " ", value)
    return re.sub(r"\s+", " ", value).strip(" :-.")


def _is_name_value(value: str | None) -> bool:
    if not value or len(value) < 2:
        return False
    token = value.upper().strip()
    if any(word in _FIELD_WORDS for word in token.split()):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z\s'.-]{1,50}", token))


def _label(text: str, labels: str) -> str | None:
    match = re.search(
        rf"(?:{labels})(?:[^\S\n]*/[^\S\n]*[A-Za-z\u0900-\u097F][^\n:]*)?[^\S\n]*[:.\-]?[^\S\n]*([^\n|]*)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = _strip_bilingual(match.group(1))
    value = re.split(
        r"\s{2,}|(?:\s+DATE\s+)|(?:\s+SEX\s+)|(?:\s+EXPIR)|(?:\s+DOI\b)|(?:\s+DOB\b)",
        value,
        maxsplit=1,
    )[0].strip(" :-.")
    if value and value.upper() not in _FIELD_WORDS:
        return value
    rest = text[match.end() :]
    for line in rest.splitlines():
        cleaned = _strip_bilingual(line)
        if not cleaned:
            continue
        if cleaned.upper().split()[0] in _FIELD_WORDS:
            break
        return cleaned
    return value or None


def _repair_ocr_date_str(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip().lstrip("=*~_>| ]")
    m = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", raw)
    if not m:
        return raw
    d_str, m_str, y_str = m.group(1), m.group(2), m.group(3)
    d = int(d_str)
    if d > 31 and (d_str.startswith("4") or d_str.startswith("7")):
        d = 10 + int(d_str[1])
        return f"{d:02d}/{m_str}/{y_str}"
    return raw


def _collect_dates(text: str) -> list[str]:
    upper = text.upper()
    found: list[str] = []
    seen: set[str] = set()
    raw_candidates = (*P.DATE_DMY.findall(upper), *_LOOSE_DMY.findall(upper), *P.DATE_MON.findall(upper))
    for raw in raw_candidates:
        parsed = P.parse_date(raw) or P.parse_date(_repair_ocr_date_str(raw))
        if not parsed:
            continue
        token = parsed.strftime("%d/%m/%Y")
        if token not in seen:
            seen.add(token)
            found.append(token)
    # Also search for impossible day patterns like 42/09/2005 directly
    for loose in re.findall(r"\b([47]\d[./-]\d{1,2}[./-]\d{4})\b", upper):
        repaired = _repair_ocr_date_str(loose)
        parsed = P.parse_date(repaired)
        if parsed:
            token = parsed.strftime("%d/%m/%Y")
            if token not in seen:
                seen.add(token)
                found.append(token)
    return found


def _as_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = P.parse_date(value) or P.parse_date(_repair_ocr_date_str(value))
    if parsed:
        return parsed.strftime("%d/%m/%Y")
    dates = _collect_dates(value)
    return dates[0] if dates else None


def _dates_after_label(text: str, labels: str) -> list[str]:
    match = re.search(rf"(?:{labels})(.{{0,160}})", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return _collect_dates(match.group(1))


def _latest_unused_date(dates: list[str], *exclude: str | None) -> str | None:
    blocked = {value for value in exclude if value}
    candidates = [value for value in dates if value not in blocked]
    if not candidates:
        return None
    return max(candidates, key=lambda value: P.parse_date(value).timestamp() if P.parse_date(value) else 0)


def _to_yymmdd(value: str | None) -> str | None:
    parsed = P.parse_date(value)
    if not parsed:
        return None
    return parsed.strftime("%y%m%d")


_OCR_TO_DIGIT = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
})
_DIGIT_CONFUSIONS = {
    "0": set("0OQDC"),
    "1": set("1IL|"),
    "2": set("2Z"),
    "3": set("3589B"),
    "4": set("4A"),
    "5": set("5S36"),
    "6": set("6G5"),
    "7": set("7T1"),
    "8": set("8B3"),
    "9": set("93"),
}


def _clean_mrz_line(line: str) -> str:
    line = line.upper().replace("«", "<").replace("‹", "<").replace("»", "<")
    line = line.replace(" ", "")
    return re.sub(r"[^A-Z0-9<]", "", line)


def _ocr_as_digit(ch: str) -> str:
    if ch.isdigit():
        return ch
    if ch == "<":
        return ch
    return ch.translate(_OCR_TO_DIGIT)


def _check_is_ocr_of(expected: str, actual: str) -> bool:
    return actual in _DIGIT_CONFUSIONS.get(expected, {expected})


def _fix_checked_digits(data: str, check: str) -> tuple[str, str]:
    digits = "".join(_ocr_as_digit(ch) if not ch.isdigit() else ch for ch in data)
    check_fixed = _ocr_as_digit(check) if check else check
    if P.mrz_check_digit(digits, check_fixed):
        return digits, check_fixed
    expected = P.mrz_compute(digits) if digits else ""
    if check_fixed and expected and _check_is_ocr_of(expected, check_fixed):
        return digits, expected
    if digits.isdigit() and check_fixed and str(check_fixed).isdigit():
        for index, ch in enumerate(digits):
            for alt in _DIGIT_CONFUSIONS.get(ch, {ch}):
                if alt == ch or not alt.isdigit():
                    continue
                trial = digits[:index] + alt + digits[index + 1 :]
                if P.mrz_check_digit(trial, check_fixed):
                    return trial, check_fixed
    return (digits if digits.isdigit() else data), (check_fixed or check)


def _apply_visual_checked_date(field: str, check: str, visual_date: str | None) -> tuple[str, str]:
    visual = _to_yymmdd(visual_date)
    if visual and field == visual:
        expected = P.mrz_compute(visual)
        use_check = check if P.mrz_check_digit(visual, check) else expected
        return visual, use_check
    if P.mrz_check_digit(field, check):
        return field, check
    if visual:
        expected = P.mrz_compute(visual)
        use_check = check if P.mrz_check_digit(visual, check) else expected
        return visual, use_check
    field, check = _fix_checked_digits(field, check)
    return field, check


def _normalize_doc_number_prefix(line: str) -> str:
    if len(line) < 8:
        return line
    chars = list(line)
    letter_count = 2 if chars[0].isalpha() and len(chars) > 1 and chars[1].isalpha() else 1
    if not chars[0].isalpha():
        letter_count = 0
    for index in range(letter_count, min(8, len(chars))):
        if not chars[index].isdigit():
            chars[index] = _ocr_as_digit(chars[index])
    return "".join(chars)


def _insert_doc_number_filler(line: str) -> str:
    if len(line) < 8:
        return line
    if P.passport_number_ok(line[:8]):
        if len(line) == 8:
            return line + "<"
        eighth = line[8]
        if eighth == "<":
            return line
        if eighth in "CKEOQ":
            return line[:8] + "<" + line[9:]
        if eighth.isdigit() and len(line) < 44:
            return line[:8] + "<" + line[8:]
    return line


def _fix_nationality(line: str) -> str:
    if len(line) < 13:
        return line
    token = line[10:13]
    compact = token.replace("<", "")
    guess = compact.translate(str.maketrans({"1": "I", "0": "D", "Q": "D"}))
    if guess in {"IND", "IN"} or token in {"1ND", "IN0", "IMD", "JND", "IND"}:
        return line[:10] + "IND" + line[13:]
    return line


def _fix_mrz_dates_and_sex(line: str) -> str:
    if len(line) < 28:
        return line
    chars = list(line)
    for index in list(range(13, 20)) + list(range(21, 28)):
        chars[index] = _ocr_as_digit(chars[index])
    if chars[20] not in "MF<":
        chars[20] = {"H": "M", "N": "M", "P": "F", "E": "F"}.get(chars[20], chars[20])
    for index in range(28, min(42, len(chars))):
        if chars[index] in "CKEOQ":
            chars[index] = "<"
    return "".join(chars)


def _apply_visual_doc_number(line: str, visual_number: str | None) -> str:
    field = P.to_mrz_doc_number(visual_number or "")
    if not field or len(line) < 10:
        return line
    check = _ocr_as_digit(line[9])
    current = line[0:9]
    if P.mrz_check_digit(current, check) and P.passport_number_ok(current.replace("<", "")):
        return line
    if P.mrz_check_digit(field, check):
        return field + check + line[10:]
    expected = P.mrz_compute(field)
    if _check_is_ocr_of(expected, line[9]) or not str(line[9]).isdigit():
        return field + expected + line[10:]
    if not P.mrz_check_digit(current, check):
        return field + expected + line[10:]
    return line


def _repair_td3_line1(line: str, visual: dict | None = None) -> str:
    visual = visual or {}
    if line.startswith("PIND"):
        line = "P<" + line[1:]
    if line.startswith("PKIND") or line.startswith("PRIND"):
        line = "P<IND" + line[5:]
    if line.startswith("P<1ND"):
        line = "P<IND" + line[5:]
    v_given = visual.get("given_names", "")
    if v_given:
        for word in v_given.split():
            if len(word) >= 4 and word.endswith("O"):
                misread = word[:-1] + "D"
                if misread in line:
                    line = line.replace(misread, word)
    # Repair chevron noise in the filler: any cluster of chevrons mixed with misread letters at the end
    line = re.sub(r"(?<=<)[KCEFX0]{1,3}(?=<|$)", lambda m: "<" * len(m.group(0)), line)
    line = re.sub(r"[<KCEFX0]{4,}$", lambda m: re.sub(r"[KCEFX0]", "<", m.group(0)), line)
    if line.startswith("P") and 20 <= len(line) < 44:
        line = line.ljust(44, "<")
    return line[:44] if len(line) > 44 else line


def _repair_td3_line2(line: str, visual: dict | None = None) -> str:
    visual = visual or {}
    line = _normalize_doc_number_prefix(line)
    line = _insert_doc_number_filler(line)
    line = _fix_nationality(line)
    if 20 <= len(line) < 44:
        line = line.ljust(44, "<")
    line = line[:44]
    line = _fix_mrz_dates_and_sex(line)
    line = _apply_visual_doc_number(line, visual.get("passport_number"))
    if len(line) != 44:
        return line
    number, num_cd = line[0:9], _ocr_as_digit(line[9])
    if not P.mrz_check_digit(number, num_cd):
        expected = P.mrz_compute(number)
        if _check_is_ocr_of(expected, num_cd) or not str(line[9]).isdigit():
            num_cd = expected
    dob, dob_cd = _apply_visual_checked_date(line[13:19], line[19], visual.get("date_of_birth"))
    expiry, exp_cd = _apply_visual_checked_date(line[21:27], line[27], visual.get("date_of_expiry"))
    optional, opt_cd = _fix_checked_digits(line[28:42], line[42])
    rebuilt = number + num_cd + line[10:13] + dob + dob_cd + line[20] + expiry + exp_cd + optional + opt_cd
    if len(rebuilt) == 43:
        composite = line[43] if len(line) > 43 else P.mrz_compute(rebuilt[0:10] + rebuilt[13:20] + rebuilt[21:43])
        expected = P.mrz_compute(rebuilt[0:10] + rebuilt[13:20] + rebuilt[21:43])
        if _check_is_ocr_of(expected, composite) or not composite.isdigit():
            composite = expected
        rebuilt += composite
    return rebuilt[:44]


def _looks_like_mrz_line(line: str) -> bool:
    compact = _clean_mrz_line(line)
    if re.match(r"P<?IND", compact) and len(compact) >= 20:
        return True
    if compact.startswith("P<") and compact.count("<") >= 2 and len(compact) >= 28:
        return True
    prefix = _normalize_doc_number_prefix(compact)
    return bool(
        re.match(r"([A-Z][0-9]{7}|[A-Z]{2}[0-9]{6})", prefix)
        and len(compact) >= 20
        and (compact.count("<") >= 1 or "IND" in compact[6:16] or len(compact) >= 28)
    )


def _split_visual_and_mrz(text: str) -> tuple[str, str]:
    visual: list[str] = []
    mrz: list[str] = []
    for line in text.splitlines():
        if _looks_like_mrz_line(line):
            mrz.append(line)
        else:
            visual.append(line)
    return "\n".join(visual), "\n".join(mrz)


def _score_line1_candidate(line: str, visual: dict) -> float:
    repaired = _repair_td3_line1(line, visual)
    score = 0.0
    if repaired.startswith("P<IND"):
        score += 30.0
    elif repaired.startswith("P<"):
        score += 15.0
    if "<<" in repaired:
        score += 20.0
    diff = abs(len(repaired) - 44)
    score += max(0.0, 20.0 - diff * 2)
    s_vis = visual.get("surname", "")
    g_vis = visual.get("given_names", "")
    if s_vis and s_vis in repaired:
        score += 25.0
    if g_vis and g_vis.split()[0] in repaired:
        score += 25.0
    return score


def _score_line2_candidate(line: str, visual: dict) -> float:
    repaired = _repair_td3_line2(line, visual)
    score = 0.0
    if len(repaired) == 44:
        score += 15.0
    checks = P.td3_line2_checks(repaired)
    for k, ok in checks.items():
        if ok:
            score += 20.0
    v_num = visual.get("passport_number", "")
    if v_num and v_num in repaired[:10]:
        score += 25.0
    v_dob = _to_yymmdd(visual.get("date_of_birth"))
    if v_dob and v_dob == repaired[13:19]:
        score += 20.0
    v_exp = _to_yymmdd(visual.get("date_of_expiry"))
    if v_exp and v_exp == repaired[21:27]:
        score += 20.0
    return score


def _td3_lines(mrz_text: str, visual: dict | None = None) -> tuple[str | None, str | None]:
    visual = visual or {}
    cleaned_lines = [_clean_mrz_line(line) for line in mrz_text.splitlines()]
    cleaned_lines = [line for line in cleaned_lines if len(line) >= 10]
    blob = _clean_mrz_line(mrz_text)
    blob = re.sub(r"^PIND", "P<IND", blob)
    
    l1_cands = [
        l for l in cleaned_lines
        if re.match(r"P[<A-Z0-9]", l) and len(l) >= 20 and ("IND" in l or "<<" in l or l.startswith("P<"))
    ]
    l2_cands = [
        l for l in cleaned_lines
        if l not in l1_cands and re.match(r"[A-Z]{1,2}[0-9OILZSBG]{6,7}", l) and len(l) >= 20
    ]
    
    line1 = max(l1_cands, key=lambda l: _score_line1_candidate(l, visual)) if l1_cands else None
    line2 = max(l2_cands, key=lambda l: _score_line2_candidate(l, visual)) if l2_cands else None
    
    if line1 is None:
        match = re.search(r"P<?IND[A-Z<]{8,}", blob)
        if match:
            line1 = match.group(0)
    if line2 is None:
        match = re.search(r"([A-Z][0-9OILZSBG]{7}|[A-Z]{2}[0-9OILZSBG]{6})<?[0-9A-Z<]{18,}", blob)
        if match:
            line2 = match.group(0)
    if line1:
        line1 = _repair_td3_line1(line1, visual)
    if line2:
        line2 = _repair_td3_line2(line2, visual)
    return line1, line2


def _mrz_names(line1: str) -> tuple[str | None, str | None]:
    if len(line1) < 6:
        return None, None
    name_field = line1[5:]
    parts = name_field.split("<<", 1)
    surname = parts[0].replace("<", " ").strip() or None
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else None
    if given:
        given = re.sub(r"\s+", " ", given)
        words = given.split()
        if len(words) > 1 and len(words[-1]) == 1 and words[-1] in "KCEXF0":
            words.pop()
            given = " ".join(words)
    if surname:
        surname = re.sub(r"\s+", " ", surname)
        words = surname.split()
        if len(words) > 1 and len(words[-1]) == 1 and words[-1] in "KCEXF0":
            words.pop()
            surname = " ".join(words)
    if surname and not _is_confident_name(surname):
        surname = None
    if given and not _is_confident_name(given):
        given = None
    return surname, given


def _parse_td3(mrz_text: str, visual: dict | None = None) -> dict[str, Any]:
    """Parse ICAO 9303 TD3 (passport) MRZ, repairing common OCR substitutions."""
    line1, line2 = _td3_lines(mrz_text, visual)
    fields: dict[str, Any] = {}
    if line1 and line1.startswith("P") and len(line1) >= 10:
        country = line1[2:5].replace("<", "")
        if country:
            fields["issuing_country"] = country
        surname, given = _mrz_names(line1)
        if surname:
            fields["surname"] = surname
        if given:
            fields["given_names"] = given
        fields["mrz_line1"] = line1
    if line2 and len(line2) >= 28:
        number = line2[0:9].replace("<", "")
        if number and P.passport_number_ok(number):
            fields["passport_number"] = number
        if len(line2) >= 13:
            nationality = line2[10:13].replace("<", "")
            if nationality:
                fields["nationality"] = nationality
        dob = P.parse_mrz_date(line2[13:19]) if len(line2) >= 19 else None
        if dob:
            fields["date_of_birth"] = dob
        if len(line2) > 20 and line2[20] in "MF":
            fields["sex"] = line2[20]
        expiry = P.parse_mrz_date(line2[21:27]) if len(line2) >= 27 else None
        if expiry:
            fields["date_of_expiry"] = expiry
        fields["mrz_line2"] = line2
        fields["mrz_present"] = True
    if fields.get("given_names") or fields.get("surname"):
        fields["full_name"] = " ".join(
            part for part in (fields.get("given_names"), fields.get("surname")) if part
        )
    return fields


def _parse_mrz(text: str) -> dict[str, Any]:
    return _parse_td3(text)


def _is_confident_name(value: str | None) -> bool:
    if not value or len(value.strip()) < 2:
        return False
    token = re.sub(r"\s+", " ", value.upper()).strip()
    if any(word in _PASSPORT_JUNK or word in _FIELD_WORDS for word in token.split()):
        return False
    if re.search(r"(.)\1{3,}", token):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z\s'.-]{1,50}", token))


def _is_place_value(value: str | None) -> bool:
    if not value:
        return False
    token = re.sub(r"\s+", " ", value.upper()).strip(" ,.-")
    token = re.sub(r"\s+\d+$", "", token).strip()
    if any(word in _PASSPORT_JUNK or word in _FIELD_WORDS for word in token.replace(",", " ").split()):
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z\s,.'-]{1,50}", token))


def _labeled_line_value(text: str, labels: str, validator: Any = None) -> str | None:
    lines = [line.strip() for line in text.splitlines()]
    pattern = re.compile(rf"(?:{labels})", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        after = line[match.end():].lstrip(" :./-=")
        after = _strip_bilingual(after)
        if validator and validator(after):
            return after
        elif not validator and after and after.upper() not in _FIELD_WORDS and after.upper() not in _PASSPORT_JUNK:
            return after
        for nxt in lines[index + 1 : index + 4]:
            cleaned = _strip_bilingual(nxt.strip(" :./-="))
            if not cleaned:
                continue
            head = cleaned.upper().split()[0] if cleaned.split() else ""
            if head in _FIELD_WORDS or pattern.search(cleaned) or any(w in cleaned.upper() for w in ("NAME", "SURNAME", "DATE", "SEX", "BIRTH", "PLACE", "NATIONALITY")):
                break
            if validator and validator(cleaned):
                return cleaned
            elif not validator:
                return cleaned
    return _label(text, labels)


def _visual_passport_fields(text: str) -> dict[str, Any]:
    """Extract labelled visual-zone fields only — no headings or security-print junk."""
    fields: dict[str, Any] = {}
    number = _first(_PASSPORT_LABELED, text.upper())
    if not number:
        labeled = _labeled_line_value(text, r"PASSPORT\s*(?:NO\.?|NUMBER|#)|PASSPORT NO")
        if labeled:
            match = P.PASSPORT_NUMBER.search(re.sub(r"\s+", "", labeled.upper()))
            if not match:
                match = P.PASSPORT_NUMBER_SPACED.search(labeled.upper())
            number = re.sub(r"\s+", "", match.group(0)) if match else None
    if number and P.passport_number_ok(number):
        fields["passport_number"] = number

    surname = _labeled_line_value(text, r"SURNAME|FAMILY NAME", _is_confident_name)
    given = _labeled_line_value(text, r"GIVEN NAME\(S\)|GIVEN NAMES?|FORENAMES?", _is_confident_name)
    if _is_confident_name(surname):
        fields["surname"] = surname.upper()
    if _is_confident_name(given):
        fields["given_names"] = given.upper()

    nationality = _labeled_line_value(text, r"NATIONALITY|COUNTRY CODE")
    if nationality:
        token = re.sub(r"[^A-Z]", "", nationality.upper())
        if token in {"IND", "INDIAN"} or token.startswith("IND"):
            fields["nationality"] = "IND"
            fields["issuing_country"] = "IND"

    sex_value = _labeled_line_value(text, r"SEX|GENDER")
    if sex_value:
        token = sex_value.strip().upper()[:1]
        if token in "MF":
            fields["sex"] = token
    else:
        match = re.search(r"\bSEX\b[^\n]{0,20}\b([MF])\b", text.upper())
        if match:
            fields["sex"] = match.group(1)

    dob = (
        _as_date(_labeled_line_value(text, r"DATE OF BIRTH|D\.?O\.?B\.?"))
        or (_dates_after_label(text, r"DATE OF BIRTH|D\.?O\.?B\.?") or [None])[0]
    )
    issue_dates = _dates_after_label(text, r"DATE OF ISSUE|ISSUED ON|D\.?O\.?I\.?")
    expiry_dates = _dates_after_label(text, _EXPIRY_LABELS)
    issue = _as_date(_labeled_line_value(text, r"DATE OF ISSUE|ISSUED ON|D\.?O\.?I\.?"))
    if issue_dates:
        issue = issue_dates[0]
    expiry = _as_date(_labeled_line_value(text, _EXPIRY_LABELS))
    if expiry_dates:
        expiry = expiry_dates[-1]

    pair = _TWO_DATES.search(re.sub(r"[\u0900-\u097F]+", " ", text.upper()))
    if pair:
        first, second = _as_date(pair.group(1)), _as_date(pair.group(2))
        if first and second and first != second and first != dob and second != dob:
            dt1, dt2 = P.parse_date(first), P.parse_date(second)
            if dt1 and dt2:
                if dt1 < dt2:
                    issue, expiry = first, second
                else:
                    issue, expiry = second, first
        elif first and not issue and first != dob:
            issue = first
        elif second and (not expiry or expiry == issue) and second not in {dob, first}:
            expiry = second

    if not expiry or expiry == issue or (issue and expiry and P.parse_date(issue) and P.parse_date(expiry) and P.parse_date(issue) > P.parse_date(expiry)):
        all_d = _collect_dates(text)
        non_dob = [d for d in all_d if d != dob]
        parsed_candidates = sorted([(P.parse_date(d), d) for d in non_dob if P.parse_date(d)], key=lambda x: x[0].timestamp())
        if len(parsed_candidates) >= 2:
            issue = parsed_candidates[0][1]
            expiry = parsed_candidates[-1][1]
        elif parsed_candidates:
            expiry = parsed_candidates[-1][1]

    if issue and expiry:
        dt_i, dt_e = P.parse_date(issue), P.parse_date(expiry)
        if dt_i and dt_e and dt_i > dt_e:
            issue, expiry = expiry, issue

    if dob:
        fields["date_of_birth"] = dob
    if issue and issue != dob:
        fields["date_of_issue"] = issue
    if expiry and expiry != dob:
        fields["date_of_expiry"] = expiry

    pob = _labeled_line_value(text, r"PLACE OF BIRTH", _is_place_value)
    poi = _labeled_line_value(text, r"PLACE OF ISSUE", _is_place_value)
    if _is_place_value(pob):
        pob_clean = re.sub(r"\s+\d+$", "", pob.upper()).strip()
        fields["place_of_birth"] = re.sub(r"\s+", " ", pob_clean)
    if _is_place_value(poi):
        poi_clean = re.sub(r"\s+\d+$", "", poi.upper()).strip()
        fields["place_of_issue"] = re.sub(r"\s+", " ", poi_clean)

    if fields.get("given_names") or fields.get("surname"):
        fields["full_name"] = " ".join(
            part for part in (fields.get("given_names"), fields.get("surname")) if part
        )
    return {k: v for k, v in fields.items() if v}


def _pick_field(mrz: dict[str, Any], visual: dict[str, Any], key: str, trusted: bool) -> Any:
    if trusted and mrz.get(key):
        return mrz[key]
    if visual.get(key):
        return visual[key]
    return mrz.get(key)


def _extract_passport_fields(visual_text: str, mrz_text: str) -> dict[str, Any]:
    visual = _visual_passport_fields(visual_text)
    mrz = _parse_td3(mrz_text, visual)
    fields: dict[str, Any] = {"document_type": "PASSPORT"}
    checks = P.td3_line2_checks(str(mrz.get("mrz_line2") or ""))
    trusted_number = checks.get("mrz_passport_cd", False)
    trusted_dob = checks.get("mrz_dob_cd", False)
    trusted_expiry = checks.get("mrz_expiry_cd", False)
    mrz_priority = (
        "surname",
        "given_names",
        "nationality",
        "sex",
        "issuing_country",
        "mrz_line1",
        "mrz_line2",
        "mrz_present",
    )
    for key in mrz_priority:
        if mrz.get(key):
            fields[key] = mrz[key]
    if visual.get("given_names") and _is_confident_name(visual["given_names"]):
        v_g = visual["given_names"].strip()
        m_g = fields.get("given_names", "").strip()
        if not m_g or m_g == v_g or (len(m_g) == len(v_g) and sum(c1 != c2 for c1, c2 in zip(m_g, v_g)) <= 2):
            fields["given_names"] = v_g
    if visual.get("surname") and _is_confident_name(visual["surname"]):
        v_s = visual["surname"].strip()
        m_s = fields.get("surname", "").strip()
        if not m_s or m_s == v_s or (len(m_s) == len(v_s) and sum(c1 != c2 for c1, c2 in zip(m_s, v_s)) <= 2):
            fields["surname"] = v_s
    number = _pick_field(mrz, visual, "passport_number", trusted_number)
    if number:
        fields["passport_number"] = number
        fields["mrz_number"] = number
    elif visual.get("passport_number"):
        fields["passport_number"] = visual["passport_number"]
        fields["mrz_number"] = visual["passport_number"]
    dob = _pick_field(mrz, visual, "date_of_birth", trusted_dob)
    if dob:
        fields["date_of_birth"] = dob
    expiry = _pick_field(mrz, visual, "date_of_expiry", trusted_expiry)
    if expiry:
        fields["date_of_expiry"] = expiry
    for key in ("place_of_birth", "place_of_issue", "date_of_issue"):
        if visual.get(key):
            fields[key] = visual[key]
    for key in mrz_priority:
        if fields.get(key):
            continue
        value = visual.get(key)
        if not value:
            continue
        if key == "surname" and fields.get("given_names") and value == fields["given_names"]:
            continue
        fields[key] = value
    if fields.get("given_names") or fields.get("surname"):
        fields["full_name"] = " ".join(
            part for part in (fields.get("given_names"), fields.get("surname")) if part
        )
    return _clean_passport_display({k: fields[k] for k in _PASSPORT_OUTPUT_KEYS if fields.get(k) not in (None, "", {})})


def _clean_passport_display(fields: dict[str, Any]) -> dict[str, Any]:
    """Keep only characters that belong in each passport field."""
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if not isinstance(value, str):
            cleaned[key] = value
            continue
        if key in {"surname", "given_names", "full_name"}:
            token = re.sub(r"\s+", " ", re.sub(r"[^A-Z\s'-]", "", value.upper())).strip()
            if token and _is_confident_name(token):
                cleaned[key] = token
        elif key in {"place_of_birth", "place_of_issue"}:
            token = re.sub(r"\s+", " ", re.sub(r"[^A-Z\s,.'-]", "", value.upper())).strip(" ,")
            if token and _is_place_value(token):
                cleaned[key] = token
        elif key in {"mrz_line1", "mrz_line2"}:
            token = re.sub(r"[^A-Z0-9<]", "", value.upper())
            if token:
                cleaned[key] = token
        elif key in {"passport_number", "mrz_number", "nationality", "issuing_country", "sex"}:
            token = re.sub(r"[^A-Z0-9]", "", value.upper())
            if token:
                cleaned[key] = token
        else:
            cleaned[key] = value
    if cleaned.get("given_names") or cleaned.get("surname"):
        cleaned["full_name"] = " ".join(
            part for part in (cleaned.get("given_names"), cleaned.get("surname")) if part
        )
    return {k: cleaned[k] for k in _PASSPORT_OUTPUT_KEYS if cleaned.get(k) not in (None, "", {})}


def _visual_aadhaar_fields(text: str) -> dict[str, Any]:
    upper = text.upper()
    fields: dict[str, Any] = {}
    aadhaar = _first(P.AADHAAR_NUMBER, upper)
    if aadhaar:
        digits = re.sub(r"\D", "", aadhaar)
        fields["aadhaar_number"] = digits
        fields["aadhaar_display"] = f"{digits[:4]} {digits[4:8]} {digits[8:]}" if len(digits) == 12 else digits
        fields["masked"] = False
    else:
        # Check masked Aadhaar
        masked_match = P.MASKED_AADHAAR.search(upper)
        if masked_match:
            fields["aadhaar_number"] = masked_match.group(1).upper()
            fields["aadhaar_display"] = masked_match.group(1).upper()
            fields["masked"] = True
        elif "XXXX" in upper or "••••" in upper:
            last4 = re.search(r"(?:XXXX|X{4}|••••)[\s-]*(?:XXXX|X{4}|••••)?[\s-]*(\d{4})", upper)
            if last4:
                fields["aadhaar_number"] = f"XXXX XXXX {last4.group(1)}"
                fields["aadhaar_display"] = f"XXXX XXXX {last4.group(1)}"
                fields["masked"] = True

    name = _label(text, r"\bNAME\b")
    if not _is_name_value(name):
        name = None
        for line in text.splitlines():
            line = line.strip()
            if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}", line):
                name = line
                break
    if name:
        fields["full_name"] = name.upper()
    dob = _as_date(_label(text, r"DOB|DATE OF BIRTH|जन्म\s*तिथि|जन्म"))
    if not dob:
        dob_match = re.search(r"(?:DOB|जन्म\s*तिथि)[:\s]+(\d{1,2}[./-]\d{1,2}[./-]\d{4})", text, re.IGNORECASE)
        if dob_match:
            dob = _as_date(dob_match.group(1))
    if dob:
        fields["date_of_birth"] = dob
    if re.search(r"\bFEMALE\b|महिला", upper):
        fields["sex"] = "F"
    elif re.search(r"\bMALE\b|पुरुष", upper):
        fields["sex"] = "M"

    co = _label(text, r"C/O|S/O|D/O|W/O|CARE OF|आत्मज|सुपुत्र")
    if co and _is_name_value(co):
        fields["parent_or_spouse"] = co.upper()
    addr = _label(text, r"ADDRESS|पता")
    if addr and len(addr) > 5:
        fields["address"] = addr.upper()
    return {k: v for k, v in fields.items() if v not in (None, "", False)}


def _visual_dl_fields(text: str) -> dict[str, Any]:
    upper = text.upper()
    fields: dict[str, Any] = {}
    raw = _first(P.DL_NUMBER, upper)
    if not raw:
        loose = re.search(r"\b([A-Z]{2}[-\s/]?\d{2}[-\s/]?\d{4}[-\s/]?\d{5,7})\b", upper)
        if loose:
            raw = loose.group(1)
    if raw and P.dl_number_ok(raw):
        fields["dl_number"] = P.format_dl_number(raw)
        compact = re.sub(r"[^A-Z0-9]", "", raw)
        fields["issuing_state"] = compact[:2]
    name = _label(text, r"\bNAME\b")
    if _is_name_value(name):
        fields["full_name"] = name.upper()
    guardian = _label(text, r"S/D/W OF|S/O|D/O|W/O|GUARDIAN")
    if _is_name_value(guardian):
        fields["parent_or_spouse"] = guardian.upper()
    dob = _as_date(_label(text, r"\bDOB\b|DATE OF BIRTH"))
    issue = _as_date(_label(text, r"\bDOI\b|DATE OF ISSUE|ISSUED ON"))
    expiry = _as_date(_label(text, r"VALID TILL|VALID UNTIL|VALID TO|EXPIRY"))
    if dob:
        fields["date_of_birth"] = dob
    if issue:
        fields["date_of_issue"] = issue
    if expiry:
        fields["date_of_expiry"] = expiry
    blood = _label(text, r"BLOOD GROUP")
    bg = re.search(r"\b(A|B|AB|O)[+-]", (blood or upper))
    if bg:
        fields["blood_group"] = bg.group(0)
    address = _label(text, r"ADDRESS")
    if address:
        fields["address"] = address.upper()
    cov = re.findall(r"\b(MCWG|LMV|MCWOG|HGV|HPMV|LMV-NT)\b", upper)
    if cov:
        fields["vehicle_classes"] = sorted(set(cov))
    return {k: v for k, v in fields.items() if v}



def _prefer_passport_number(*candidates: str | None) -> str | None:
    valid = [c for c in candidates if c and P.passport_number_ok(c)]
    if valid:
        return valid[0]
    return next((c for c in candidates if c), None)


def _extract_visa_fields(visual_text: str, mrz_text: str = "") -> dict[str, Any]:
    text = f"{visual_text}\n{mrz_text}".strip()
    upper = text.upper()
    fields: dict[str, Any] = {"document_type": "VISA", "raw_text": text}

    # 1. Issuing Country
    if "REPUBLIC OF SINGAPORE" in upper or "SINGAPORE" in upper:
        fields["issuing_country"] = "REPUBLIC OF SINGAPORE"
    elif "REPUBLIC OF INDIA" in upper:
        fields["issuing_country"] = "REPUBLIC OF INDIA"
    elif "UNITED STATES OF AMERICA" in upper or "UNITED STATES" in upper:
        fields["issuing_country"] = "UNITED STATES OF AMERICA"
    elif "UNITED KINGDOM" in upper:
        fields["issuing_country"] = "UNITED KINGDOM"

    # 2. Travel Document No / Passport No
    doc_m = re.search(
        r"(?:TRAVEL\s*DOCUMENT\s*(?:NO\.?|NUMBER|#)?|PASSPORT\s*(?:NO\.?|NUMBER|#)?)[^\w\n]*([A-Z]{1,2}\s*[0-9]{6,7})",
        upper,
    )
    if doc_m:
        cand_pass = re.sub(r"\s+", "", doc_m.group(1))
        if P.passport_number_ok(cand_pass):
            fields["passport_number"] = cand_pass
            fields["travel_document_number"] = cand_pass
    if not fields.get("passport_number"):
        any_p = P.PASSPORT_NUMBER.search(upper)
        if any_p:
            fields["passport_number"] = any_p.group(1)
            fields["travel_document_number"] = any_p.group(1)

    # 3. Visa / Sticker Number
    v_m = re.search(
        r"(?:VISA\s*(?:NO\.?|NUMBER|#)?)\s*[:.\-]?\s*([A-Z]{1,3}\s*[0-9]{6,8}|[0-9]{7,8})",
        upper,
    )
    if v_m:
        fields["visa_number"] = v_m.group(1).strip()
    if not fields.get("visa_number"):
        for cand in re.findall(r"\b([A-Z]{1,2}\s*[0-9]{7}|[0-9]{7,8})\b", upper):
            cl = re.sub(r"\s+", "", cand)
            if cl != fields.get("passport_number"):
                fields["visa_number"] = cand.strip()
                break
    if not fields.get("visa_number"):
        cand = P.VISA_NUMBER.search(upper)
        if cand and re.sub(r"\s+", "", cand.group(1)) != fields.get("passport_number"):
            fields["visa_number"] = cand.group(1).strip()

    # 4. Full Name, Surname, Given Names
    name_m = re.search(r"\b(?:NAME|BEARER|SURNAME|HOLDER)\s*[:.\-_/]?\s*([A-Z\s]{4,50})", upper)
    if name_m:
        raw = name_m.group(1).splitlines()[0]
        tokens = [
            t for t in re.sub(r"[^A-Z\s]", " ", raw).split()
            if len(t) >= 3 and t not in (
                "DOB", "SEX", "MALE", "FEMALE", "NATIONALITY", "PASSPORT", "TRAVEL", "DOCUMENT",
                "SPECIMEN", "VISA", "REPUBLIC", "INDIA", "SINGAPORE", "VALID", "ENTRY"
            )
        ]
        if 2 <= len(tokens) <= 5:
            fields["full_name"] = " ".join(tokens)
            fields["surname"] = tokens[0]
            fields["given_names"] = " ".join(tokens[1:])
    if not fields.get("full_name") and mrz_text:
        for line in mrz_text.splitlines():
            line_c = _clean_mrz_line(line)
            mrv_m = re.match(r"^V<([A-Z]{3})([A-Z<]{10,})", line_c)
            if mrv_m:
                name_part = mrv_m.group(2)
                if "<<" in name_part:
                    sur, given = name_part.split("<<", 1)
                    s_clean = sur.replace("<", " ").strip()
                    g_clean = given.replace("<", " ").strip()
                    if s_clean and g_clean:
                        fields["surname"] = s_clean
                        fields["given_names"] = g_clean
                        fields["full_name"] = f"{s_clean} {g_clean}"
                        break

    # 5. Date of Birth
    dob_m = re.search(r"\bDOB[^\w\n]*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", upper)
    if dob_m:
        fields["date_of_birth"] = P.normalize_date_str(dob_m.group(1))
    else:
        m_19 = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-]19\d{2})\b", upper)
        if m_19:
            fields["date_of_birth"] = P.normalize_date_str(m_19.group(1))

    # 6. Sex / Gender
    sex_m = (
        re.search(r"\b(?:SEX|GENDER)[^A-Za-z0-9\n]*([MF])\b", upper)
        or re.search(r"\b([MF])\b[^A-Za-z0-9\n]*(?:NATIONALITY|IND)", upper)
        or re.search(r"\b(MALE|FEMALE)\b", upper)
    )
    if sex_m:
        val = sex_m.group(1).upper()
        fields["sex"] = "M" if val in ("M", "MALE") else "F"
        fields["gender"] = fields["sex"]

    # 7. Nationality
    nat_m = re.search(r"\bNATIONALITY[^\w\n]*([A-Z]{3})\b", upper)
    if nat_m:
        fields["nationality"] = nat_m.group(1)
    elif "IND" in upper:
        fields["nationality"] = "IND"

    # 8. Validity Dates (Issue & Expiry)
    im = re.search(r"(?:VISA\s*ISSUE\s*DATE|ISSUE\s*DATE|DATE\s*OF\s*ISSUE)\s*[:.\-_]*\s*([0-9A-Z\s./-]+)", upper)
    if im:
        c = P.DATE_MON.search(im.group(1)) or P.DATE_DMY.search(im.group(1))
        if c:
            fields["valid_from"] = P.normalize_date_str(c.group(1))
            fields["date_of_issue"] = fields["valid_from"]

    em = re.search(r"(?:VISA\s*VALID\s*TILL|VALID\s*TILL|VALID\s*UNTIL|DATE\s*OF\s*EXPIRY|EXPIRY\s*DATE)\s*[:.\-_]*\s*([0-9A-Z\s./-]+)", upper)
    if em:
        c = P.DATE_MON.search(em.group(1)) or P.DATE_DMY.search(em.group(1))
        if c:
            fields["valid_until"] = P.normalize_date_str(c.group(1))
            fields["date_of_expiry"] = fields["valid_until"]

    # Fallback: collect non-DOB dates chronologically
    dob_norm = fields.get("date_of_birth")
    cand_dates = []
    for m in P.DATE_MON.finditer(upper):
        d = P.normalize_date_str(m.group(1))
        if d and d not in cand_dates and d != dob_norm:
            cand_dates.append(d)
    for m in P.DATE_DMY.finditer(upper):
        d = P.normalize_date_str(m.group(1))
        if d and d not in cand_dates and d != dob_norm:
            cand_dates.append(d)

    date_objs = [(P.parse_date(d), d) for d in cand_dates if P.parse_date(d)]
    date_objs.sort(key=lambda x: x[0])
    sorted_d = [x[1] for x in date_objs]

    if not fields.get("valid_from") and sorted_d:
        fields["valid_from"] = sorted_d[0]
        fields["date_of_issue"] = sorted_d[0]
    if not fields.get("valid_until") and len(sorted_d) > 1:
        fields["valid_until"] = sorted_d[-1]
        fields["date_of_expiry"] = sorted_d[-1]

    # 9. Visa Type, Entries, Period of Stay
    if "MULTIPLE JOURNEY" in upper:
        fields["visa_type"] = "MULTIPLE JOURNEY"
        fields["entries"] = "MULTIPLE"
    elif "SINGLE JOURNEY" in upper:
        fields["visa_type"] = "SINGLE JOURNEY"
        fields["entries"] = "SINGLE"
    elif "TOURIST" in upper:
        fields["visa_type"] = "TOURIST"
    elif "BUSINESS" in upper:
        fields["visa_type"] = "BUSINESS"
    elif "STUDENT" in upper:
        fields["visa_type"] = "STUDENT"
    elif "CONFERENCE" in upper:
        fields["visa_type"] = "CONFERENCE"
    elif "ENTRY" in upper and "ENTRY CLEARANCE" not in upper:
        fields["visa_type"] = "ENTRY"

    if not fields.get("entries"):
        if "MULTIPLE" in upper or "MULT" in upper:
            fields["entries"] = "MULTIPLE"
        elif "SINGLE" in upper:
            fields["entries"] = "SINGLE"
        elif "DOUBLE" in upper:
            fields["entries"] = "DOUBLE"

    if "SHORT VISIT" in upper:
        fields["period_of_stay"] = "SHORT VISIT"

    # 10. Remarks
    if "NOT VALID FOR EMPLOYMENT" in upper or ("EMPLOYMENT" in upper and "NOT" in upper):
        fields["remarks"] = "Not Valid for Employment"
    else:
        rem_m = re.search(r"\bREMARKS?\s*[:.\-]?\s*([A-Z\s,.-]+)", upper)
        if rem_m:
            rline = rem_m.group(1).splitlines()[0].strip()
            fields["remarks"] = rline.title() if rline.isupper() else rline

    # 11. Issuing Post / Authority
    auth_m = re.search(r"\b(SINGAPORE\s*CONSULATE\s*IN\s*[A-Z]+)\b", upper)
    if auth_m:
        fields["issuing_post"] = auth_m.group(1).replace("MUMBAL", "MUMBAI")

    # 12. Parse MRV if genuine V< line present
    clean_mrz = _clean_mrz_line(mrz_text)
    if "V<" in clean_mrz:
        fields["mrz_present"] = True
        fields["mrz_raw"] = mrz_text

    return {k: v for k, v in fields.items() if v not in (None, "", {})}


def extract_fields(document_type: DocumentType, text: str) -> dict[str, Any]:
    upper = text.upper()
    dates = _collect_dates(text)

    if document_type == DocumentType.PASSPORT:
        visual_text, mrz_text = _split_visual_and_mrz(text)
        blob = _clean_mrz_line(text)
        if not mrz_text.strip() and (
            re.search(r"P<?IND", blob)
            or re.search(r"([A-Z]{1,2}[0-9]{6,7})<?[0-9]IND\d{6}", blob)
        ):
            mrz_text = text
        return _extract_passport_fields(visual_text, mrz_text)

    if document_type == DocumentType.VISA:
        return _extract_visa_fields(text)

    fields: dict[str, Any] = {"raw_text": text.strip()}
    if document_type == DocumentType.NATIONAL_ID:
        fields.update(_visual_aadhaar_fields(text))
    elif document_type == DocumentType.DRIVING_LICENSE:
        fields.update(_visual_dl_fields(text))
    elif document_type == DocumentType.GENERIC:
        fields["detected_dates"] = dates
        for pat, key in [
            (P.PASSPORT_NUMBER, "possible_passport_number"),
            (P.AADHAAR_NUMBER, "possible_aadhaar_number"),
            (P.DL_NUMBER, "possible_dl_number"),
            (P.VISA_NUMBER, "possible_visa_number"),
        ]:
            match = pat.search(upper)
            if match:
                fields[key] = match.group(1)

    if dates and "date_of_birth" not in fields and document_type not in {
        DocumentType.PASSPORT,
        DocumentType.DRIVING_LICENSE,
        DocumentType.VISA,
        DocumentType.GENERIC,
    }:
        fields["date_of_birth"] = dates[0]
    if any(
        token in upper
        for token in (
            "GOVERNMENT OF INDIA",
            "REPUBLIC OF INDIA",
            "UNION OF INDIA",
            "AADHAAR",
        )
    ):
        fields["issuer_india"] = True
    return {k: v for k, v in fields.items() if v not in (None, "", {})}


def detect_document_type(text: str) -> DocumentType:
    """Classify a document into Passport, Visa, National ID (Aadhaar), DL, or Generic based on text."""
    upper = text.upper()
    blob = _clean_mrz_line(text)

    # 1. Genuine Passport TD3 MRZ (P<IND or P<[A-Z]{3})
    has_passport_mrz = "P<IND" in blob or bool(re.search(r"P<[A-Z]{3}", blob))

    # 2. Strong Visa indicators
    visa_keywords = (
        "TYPE OF VISA",
        "VISA ISSUE DATE",
        "VISA VALID TILL",
        "PERIOD OF STAY",
        "VALID FOR STAY",
        "VISA NO",
        "VISA NUMBER",
        "VISA SPECIMEN",
        "REPUBLIC OF SINGAPORE",
        "SINGAPORE VISA",
        "ENTRY CLEARANCE",
        "VISA",
    )
    has_visa = any(k in upper for k in visa_keywords) or bool(re.search(r"V<[A-Z]{3}", blob))

    # Visas frequently mention passports in conditions or labels; if strong visa indicators exist
    # and no genuine passport MRZ line is present, it is classified as a Visa.
    if has_visa and not has_passport_mrz:
        return DocumentType.VISA

    # 3. Passport
    if (
        has_passport_mrz
        or "PASSPORT" in upper
        or bool(re.search(r"\bTYPE\s*[/:\s]*P\b", upper))
        or ("REPUBLIC OF INDIA" in upper and bool(P.PASSPORT_NUMBER.search(upper)) and "VISA" not in upper)
    ):
        return DocumentType.PASSPORT

    if has_visa:
        return DocumentType.VISA

    # 3. National ID (Aadhaar)
    if (
        "AADHAAR" in upper
        or "UIDAI" in upper
        or "UNIQUE IDENTIFICATION" in upper
        or "MERA AADHAAR" in upper
        or "मेरा आधार" in text
        or "आधार" in text
        or bool(P.MASKED_AADHAAR.search(upper))
        or ("GOVERNMENT OF INDIA" in upper and bool(P.AADHAAR_NUMBER.search(upper)))
    ):
        return DocumentType.NATIONAL_ID

    # 4. Driving Licence
    if (
        "DRIVING LICEN" in upper
        or "DRIVING LICENCE" in upper
        or "DRIVING LICENSE" in upper
        or "MOTOR DRIVING" in upper
        or "MOTOR VEHICLE" in upper
        or "FORM 7" in upper
        or "DL NO" in upper
        or "DL NUMBER" in upper
        or bool(P.DL_NUMBER.search(upper))
    ):
        return DocumentType.DRIVING_LICENSE

    return DocumentType.GENERIC


def classify_document(
    image_bytes: bytes, text_hint: str | None = None
) -> tuple[DocumentType, str, float]:
    """Auto-detect document type by scanning image text and patterns."""
    if text_hint:
        doc_type = detect_document_type(text_hint)
        return doc_type, text_hint, 85.0
    image = _to_pil(image_bytes)
    text, conf = _read_text(image, None)
    doc_type = detect_document_type(text)
    if doc_type == DocumentType.GENERIC and not text.strip():
        # Fallback to passport regions
        vis, mrz, pconf = _read_passport_regions(image)
        combined = f"{vis}\n{mrz}".strip()
        if combined:
            p_type = detect_document_type(combined)
            if p_type != DocumentType.GENERIC:
                return p_type, combined, pconf
    return doc_type, text, conf


def run_ocr(
    document_type: DocumentType,
    image_bytes: bytes,
    text_hint: str | None = None,
) -> tuple[ModuleResult, list[Finding]]:
    image = _to_pil(image_bytes)
    findings: list[Finding] = []

    if document_type == DocumentType.PASSPORT:
        visual_text, mrz_text, ocr_conf = _read_passport_regions(image)
        if text_hint:
            hint_visual, hint_mrz = _split_visual_and_mrz(text_hint)
            visual_text = f"{visual_text}\n{hint_visual}".strip()
            mrz_text = f"{mrz_text}\n{hint_mrz}".strip() or text_hint
            ocr_conf = max(ocr_conf, 72.0)
        has_text = bool(visual_text.strip() or mrz_text.strip())
        if not has_text:
            findings.append(
                Finding(
                    module="ocr",
                    severity="medium",
                    code="OCR_ENGINE_UNAVAILABLE",
                    message=(
                        "No text extracted. Install Tesseract OCR for live extraction; "
                        "validation will still run on any typed/MRZ text found."
                    ),
                )
            )
            fields: dict[str, Any] = {"document_type": "PASSPORT"}
            score = 35.0
            summary = "OCR engine not available or image is unreadable."
        else:
            fields = _extract_passport_fields(visual_text, mrz_text)
            needs_fallback = not fields.get("date_of_expiry") or not fields.get("passport_number")
            if needs_fallback:
                full_text, full_conf = _read_text(image, DocumentType.PASSPORT)
                if full_text.strip():
                    hint_visual, hint_mrz = _split_visual_and_mrz(full_text)
                    visual_text = f"{visual_text}\n{hint_visual}".strip()
                    mrz_text = f"{mrz_text}\n{hint_mrz}".strip() or mrz_text
                    ocr_conf = max(ocr_conf, full_conf)
                    fields = _extract_passport_fields(visual_text, mrz_text)
            missing = [] if fields.get("passport_number") else ["passport_number"]
            identity_keys = ("surname", "given_names", "full_name", "date_of_birth", "date_of_expiry")
            filled = sum(1 for key in identity_keys if fields.get(key))
            score = min(100.0, ocr_conf + (15 if not missing else 0) + filled * 3)
            if missing:
                findings.append(
                    Finding(
                        module="ocr",
                        severity="high",
                        code="PRIMARY_NUMBER_MISSING",
                        message=f"Could not extract required field(s): {', '.join(missing)}.",
                    )
                )
                score = min(score, 45)
            summary = f"Extracted {len(fields)} fields with {ocr_conf:.0f}% OCR confidence."
        result = ModuleResult(
            name="ocr",
            score=round(min(score, 100.0), 1),
            confidence=round(ocr_conf if has_text else 20.0, 1),
            summary=summary,
            details=fields,
        )
        return result, findings

    if document_type == DocumentType.VISA:
        visual_text, mrz_text, ocr_conf = _read_visa_regions(image)
        if text_hint:
            visual_text = f"{visual_text}\n{text_hint}".strip()
            ocr_conf = max(ocr_conf, 72.0)
        has_text = bool(visual_text.strip() or mrz_text.strip())
        if not has_text:
            findings.append(
                Finding(
                    module="ocr",
                    severity="medium",
                    code="OCR_ENGINE_UNAVAILABLE",
                    message=(
                        "No text extracted. Install Tesseract OCR for live extraction; "
                        "validation will still run on any typed/MRZ text found."
                    ),
                )
            )
            fields = {"document_type": "VISA", "raw_text": ""}
            score = 35.0
            summary = "OCR engine not available or image is unreadable."
        else:
            fields = _extract_visa_fields(visual_text, mrz_text)
            fields["image_size"] = list(image.size)
            missing = [] if fields.get("visa_number") else ["visa_number"]
            identity_keys = ("surname", "given_names", "full_name", "date_of_birth", "passport_number", "valid_until", "valid_from")
            filled = sum(1 for key in identity_keys if fields.get(key))
            score = min(100.0, ocr_conf + (15 if not missing else 0) + filled * 3)
            if missing:
                findings.append(
                    Finding(
                        module="ocr",
                        severity="high",
                        code="PRIMARY_NUMBER_MISSING",
                        message=f"Could not extract required field(s): {', '.join(missing)}.",
                    )
                )
                score = min(score, 45)
            summary = f"Extracted {len(fields)} visa fields with {ocr_conf:.0f}% OCR confidence."
        result = ModuleResult(
            name="ocr",
            score=round(min(score, 100.0), 1),
            confidence=round(ocr_conf if has_text else 20.0, 1),
            summary=summary,
            details=fields,
        )
        return result, findings

    text, ocr_conf = _read_text(image, document_type)
    if text_hint:
        text = f"{text}\n{text_hint}" if text.strip() else text_hint
        ocr_conf = max(ocr_conf, 72.0)

    if not text.strip():
        findings.append(
            Finding(
                module="ocr",
                severity="medium",
                code="OCR_ENGINE_UNAVAILABLE",
                message=(
                    "No text extracted. Install Tesseract OCR for live extraction; "
                    "validation will still run on any typed/MRZ text found."
                ),
            )
        )
        fields = {
            "raw_text": "",
            "image_size": list(image.size),
        }
        score = 35.0
        summary = "OCR engine not available or image is unreadable."
    else:
        fields = extract_fields(document_type, text)
        fields["image_size"] = list(image.size)
        required_map = {
            DocumentType.VISA: ["visa_number"],
            DocumentType.NATIONAL_ID: ["aadhaar_number"],
            DocumentType.DRIVING_LICENSE: ["dl_number"],
            DocumentType.GENERIC: [],
        }
        required = required_map.get(document_type, [])
        missing = [key for key in required if not fields.get(key)]
        identity_keys = ("surname", "given_names", "full_name", "date_of_birth", "date_of_expiry")
        filled = sum(1 for key in identity_keys if fields.get(key))
        score = min(100.0, ocr_conf + (15 if not missing else 0) + filled * 3)
        if missing:
            findings.append(
                Finding(
                    module="ocr",
                    severity="high",
                    code="PRIMARY_NUMBER_MISSING",
                    message=f"Could not extract required field(s): {', '.join(missing)}.",
                )
            )
            score = min(score, 45)
        summary = (
            f"Extracted {len(fields)} fields with {ocr_conf:.0f}% OCR confidence."
        )

    result = ModuleResult(
        name="ocr",
        score=round(min(score, 100.0), 1),
        confidence=round(ocr_conf if text.strip() else 20.0, 1),
        summary=summary,
        details=fields,
    )
    return result, findings

