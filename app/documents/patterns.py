"""Field patterns and check-digit helpers for Indian identity documents."""

from __future__ import annotations

import re
from datetime import datetime, timezone

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CT", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
}

# Indian passport: 1 letter + 7 digits, or 2 letters + 6 digits.
PASSPORT_NUMBER = re.compile(r"\b([A-Z][0-9]{7}|[A-Z]{2}[0-9]{6})\b")
PASSPORT_NUMBER_SPACED = re.compile(r"\b([A-Z]{1,2})\s*([0-9]{6,7})\b")
PASSPORT_NUMBER_EXACT = re.compile(r"(?:[A-Z][0-9]{7}|[A-Z]{2}[0-9]{6})$")
AADHAAR_NUMBER = re.compile(r"\b(\d{4}\s\d{4}\s\d{4}|\d{12})\b")
MASKED_AADHAAR = re.compile(r"\b([X*•x]{4}[\s-]?[X*•x]{4}[\s-]?\d{4})\b")
DL_NUMBER = re.compile(
    r"\b([A-Z]{2}\d{2}[-\s]?\d{11}|[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{5,7}|[A-Z]{2}[-\s/]?\d{2}[-\s/]?\d{4}[-\s/]?\d{5,7})\b"
)
DATE_DMY = re.compile(r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b")
DATE_MON = re.compile(
    r"\b(\d{1,2}[-/\s](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?[-/\s]\d{2,4})\b"
)
VISA_NUMBER = re.compile(r"\b(?=[A-Z0-9]*\d)([A-Z]{1,3}\s*[0-9]{6,8}|[A-Z0-9]{7,12})\b")
ISO_NAME = re.compile(r"^[A-Z][A-Z\s'.-]{1,60}$")

# Verhoeff tables used by UIDAI Aadhaar checksum
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)

MRZ_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"


def passport_number_ok(number: str) -> bool:
    cleaned = re.sub(r"[\s-]", "", number.upper())
    return bool(PASSPORT_NUMBER_EXACT.fullmatch(cleaned))


def dl_number_ok(number: str) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", number.upper())
    if len(compact) < 13 or len(compact) > 16:
        return False
    state = compact[:2]
    if state not in INDIAN_STATES:
        return False
    if re.fullmatch(r"[A-Z]{2}\d{13}", compact):
        return True
    if re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{7,12}", compact):
        return True
    return False


def format_dl_number(number: str) -> str:
    compact = re.sub(r"[^A-Z0-9]", "", number.upper())
    if re.fullmatch(r"[A-Z]{2}\d{13}", compact):
        return f"{compact[:4]} {compact[4:]}"
    return compact


def is_masked_aadhaar(number: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(number).strip())
    return bool(MASKED_AADHAAR.search(cleaned)) or "XXXX" in cleaned.upper()


def india_code_ok(value: str | None) -> bool:
    if not value:
        return True
    token = re.sub(r"[^A-Z]", "", value.upper())
    return token in {"IND", "INDIAN"} or token.startswith("IND")


def verhoeff_check_digit(body: str) -> str:
    checksum = 0
    for i, ch in enumerate(reversed(body)):
        checksum = _D[checksum][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[checksum])


def verhoeff_ok(number: str) -> bool:
    if is_masked_aadhaar(number):
        # Masked Aadhaar has valid format (e.g. XXXX XXXX 1234)
        digits = [c for c in str(number) if c.isdigit()]
        return len(digits) == 4
    digits = [int(c) for c in re.sub(r"\D", "", str(number))]
    if len(digits) != 12:
        return False
    checksum = 0
    for i, n in enumerate(reversed(digits)):
        checksum = _D[checksum][_P[i % 8][n]]
    return checksum == 0


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"\b(MR|MRS|MS|MISS|SHRI|SMT|DR|PROF)\b\.?", "", str(name).upper())
    cleaned = re.sub(r"[^A-Z\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def name_similarity(name1: str | None, name2: str | None) -> float:
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    t1 = n1.split()
    t2 = n2.split()
    s1 = set(t1)
    s2 = set(t2)

    # 1. Exact token set equality (order independent, e.g. "KUMAR RAJESH" vs "RAJESH KUMAR")
    if s1 == s2:
        return 1.0

    # 2. Subset matching (e.g. only surname or first name written, or middle name omitted)
    if s1.issubset(s2) or s2.issubset(s1):
        shorter = s1 if len(s1) <= len(s2) else s2
        if len(shorter) >= 2:
            return 0.95
        token = next(iter(shorter))
        if len(token) >= 3:
            return 0.90
        return 0.75

    # 3. Token-level alignment with exact, initial, and fuzzy typo matching
    from difflib import SequenceMatcher

    shorter_t, longer_t = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    unused_shorter = list(shorter_t)
    unused_longer = list(longer_t)
    matched_exact = 0
    matched_initial = 0
    matched_fuzzy = 0

    # Step A: Exact matches first
    for st in list(unused_shorter):
        if st in unused_longer:
            unused_longer.remove(st)
            unused_shorter.remove(st)
            matched_exact += 1

    # Step B: Initials matching for remaining tokens
    for st in list(unused_shorter):
        if len(st) == 1:
            for lt in list(unused_longer):
                if lt.startswith(st):
                    unused_longer.remove(lt)
                    unused_shorter.remove(st)
                    matched_initial += 1
                    break
        else:
            for lt in list(unused_longer):
                if len(lt) == 1 and st.startswith(lt):
                    unused_longer.remove(lt)
                    unused_shorter.remove(st)
                    matched_initial += 1
                    break

    # Step C: Fuzzy matching for OCR typos (e.g. V vs W, O vs 0/Q)
    for st in list(unused_shorter):
        best_r = 0.0
        best_lt = None
        for lt in unused_longer:
            if len(lt) >= 3 and len(st) >= 3:
                r = SequenceMatcher(None, st, lt).ratio()
                if r > best_r:
                    best_r = r
                    best_lt = lt
        if best_r >= 0.82 and best_lt:
            unused_longer.remove(best_lt)
            unused_shorter.remove(st)
            matched_fuzzy += 1

    # If all tokens in the shorter name were accounted for:
    if not unused_shorter and (matched_exact >= 1 or len(shorter_t) == 1):
        if matched_initial > 0 or matched_fuzzy > 0:
            return 0.92
        return 0.95

    # 4. Conflicting tokens present:
    # Ensure distinct names (e.g. RAJESH KUMAR vs SURESH KUMAR) do not get boosted by character SequenceMatcher
    matched_weight = matched_exact * 1.0 + matched_initial * 0.8 + matched_fuzzy * 0.8
    token_score = matched_weight / max(len(t1), len(t2))

    intersection = s1 & s2
    union = s1 | s2
    jaccard = len(intersection) / len(union) if union else 0.0
    if jaccard >= 0.5 and matched_exact >= 2:
        return max(jaccard, 0.85)

    return min(token_score, 0.50)



def mrz_char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    return ord(ch) - 55


def mrz_compute(data: str) -> str:
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(data):
        total += mrz_char_value(ch) * weights[i % 3]
    return str(total % 10)


def mrz_check_digit(data: str, check: str) -> bool:
    return bool(check) and mrz_compute(data) == check


def to_mrz_doc_number(number: str) -> str | None:
    """Indian passport numbers are 8 characters, padded with '<' to 9 MRZ positions."""
    cleaned = re.sub(r"[\s-]", "", number.upper())
    if not passport_number_ok(cleaned):
        return None
    return (cleaned + "<" * 9)[:9]


def pad_td3_line2(line: str) -> str:
    line = re.sub(r"[^A-Z0-9<]", "", line.upper())
    if 20 <= len(line) < 44:
        line = line.ljust(44, "<")
    return line[:44] if len(line) >= 44 else line


def td3_line2_checks(line2: str) -> dict[str, bool]:
    line2 = pad_td3_line2(line2)
    if len(line2) != 44:
        return {"mrz_length": False}
    return {
        "mrz_number_format": passport_number_ok(line2[0:9].replace("<", "")),
        "mrz_passport_cd": mrz_check_digit(line2[0:9], line2[9]),
        "mrz_dob_cd": mrz_check_digit(line2[13:19], line2[19]),
        "mrz_expiry_cd": mrz_check_digit(line2[21:27], line2[27]),
        "mrz_composite": mrz_check_digit(line2[0:10] + line2[13:20] + line2[21:43], line2[43]),
    }


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    token = str(value).strip()
    loose = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", token)
    if loose:
        try:
            return datetime(int(loose.group(3)), int(loose.group(2)), int(loose.group(1)), tzinfo=timezone.utc)
        except ValueError:
            pass
    for fmt in (
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%y%m%d",
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y", "%d/%B/%Y",
        "%d-%b-%y", "%d %b %y"
    ):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_mrz_date(yymmdd: str) -> str | None:
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    year = int(yymmdd[:2])
    century = 2000 if year < 50 else 1900
    try:
        dt = datetime(century + year, int(yymmdd[2:4]), int(yymmdd[4:6]))
    except ValueError:
        return None
    return dt.strftime("%d/%m/%Y")


def normalize_date_str(value: str | None) -> str | None:
    dt = parse_date(value)
    return dt.strftime("%d/%m/%Y") if dt else None


def iso_date_str(value: str | None) -> str | None:
    dt = parse_date(value)
    return dt.strftime("%Y-%m-%d") if dt else None

