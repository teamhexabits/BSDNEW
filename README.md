# BorderGuard AI - Intelligent Travel Document Screening

Officer-assist platform for border checkpoints. It scores identity documents so staff can spend time on high-risk cases instead of reading every field by hand.

This is **decision support**, not automatic clearance. Final action stays with the officer.

## What it screens

| Document | Typical checks |
| --- | --- |
| Indian passport | MRZ, ICAO check digits, number format (1 letter + 7 digits or 2 letters + 6 digits), expiry |
| Visa | Number, validity window, optional MRZ |
| National identity (Aadhaar) | 12-digit layout, Verhoeff checksum |
| Driving licence | State code + Indian DL number pattern |

## Four modules

1. **OCR extraction** — reads the scan (Tesseract if installed) and pulls names, numbers, dates, and MRZ.
2. **Document validation** — format, checksum, date order, expiry, issuing codes.
3. **Tampering detection** — error-level analysis, recapture/blur, clone-like repeated blocks, EXIF/editor traces.
4. **Face verification** — compares the document photo with an optional live capture.

Scores are combined into an **authenticity score (0–100)** and a recommendation:

- `ALLOW` — low risk, routine release
- `SECONDARY_INSPECTION` — officer review
- `REFER_FRAUD_CELL` — high risk / multiple hard failures

## Run locally

Python 3.10+ recommended.

```powershell
cd C:\Users\Neel\Projects\border-doc-screen
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

Tesseract OCR is required for live extraction. On Windows, install [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (typically `C:\Program Files\Tesseract-OCR\tesseract.exe`). The app locates that binary on startup. **Load specimen demo** runs the same live OCR path on a labelled SPECIMEN card (not a real document).

## API

- `POST /api/screen` — `document_type`, `document` image, optional `live_photo`
- `POST /api/demo` — specimen run
- `GET /api/health`

`document_type` values: `indian_passport`, `visa`, `national_id`, `driving_license`

## Honest limits

Face match uses a center crop plus cosine/histogram similarity. It is good enough for a prototype cue, not a production biometric. Tamper cues are forensic heuristics; sophisticated forgeries can still slip through. Production deployments would add certified MRZ readers, chip/NFC (ePassport), vendor face models, and an audit log tied to the officer ID.
