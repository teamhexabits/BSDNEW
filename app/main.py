from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.demo_specimen import (
    aadhaar_specimen,
    batch_specimens,
    dl_specimen,
    passport_specimen,
    visa_specimen,
)
from app.modules.face import get_face_app, verify_faces
from app.modules.ocr import configure_ocr_engine
from app.pipeline import screen, screen_batch
from app.schemas import DocumentType, FaceVerifyResponse

ROOT = Path(__file__).resolve().parent
app = FastAPI(
    title="BorderGuard AI - Intelligent Travel Document Screening",
    description="Officer-assist scoring for passport, visa, national ID, and driving licence.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

MAX_BYTES = 8 * 1024 * 1024


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("startup")
def warmup_engines() -> None:
    tesseract_cmd = configure_ocr_engine()
    if tesseract_cmd:
        print(f"OCR engine: {tesseract_cmd}")
    else:
        print("OCR engine: Tesseract not found. Place it in .tools/tesseract or install it on PATH.")
    try:
        get_face_app()
    except Exception as exc:
        print(f"InsightFace buffalo_l failed to load at startup: {exc}")


@app.get("/api/health")
async def health():
    tesseract_cmd = configure_ocr_engine()
    return {
        "status": "ok",
        "ocr": {
            "tesseract": tesseract_cmd,
            "ready": bool(tesseract_cmd),
        },
    }


@app.post("/api/screen")
async def api_screen(
    document_type: DocumentType = Form(...),
    document: UploadFile = File(...),
    live_photo: UploadFile | None = File(None),
):
    document_bytes = await document.read()
    if not document_bytes or len(document_bytes) > MAX_BYTES:
        raise HTTPException(400, "Document image is missing or larger than 8 MB.")
    live_bytes = await live_photo.read() if live_photo and live_photo.filename else None
    if live_bytes == b"":
        live_bytes = None
    result = screen(document_type, document_bytes, live_bytes)
    return result.model_dump()


@app.post("/api/screen/batch")
async def api_screen_batch(
    documents: list[UploadFile] = File(...),
    document_types: list[str] | None = Form(None),
    live_photo: UploadFile | None = File(None),
):
    if not documents:
        raise HTTPException(400, "At least one document file must be uploaded.")

    docs_payload = []
    for idx, doc in enumerate(documents):
        content = await doc.read()
        if not content or len(content) > MAX_BYTES:
            raise HTTPException(400, f"Document '{doc.filename}' is empty or exceeds 8 MB.")
        dtype = None
        if document_types and idx < len(document_types) and document_types[idx]:
            dtype = document_types[idx]
        docs_payload.append({
            "document_bytes": content,
            "document_type": dtype,
            "filename": doc.filename or f"doc_{idx+1}.jpg",
        })

    live_bytes = await live_photo.read() if live_photo and live_photo.filename else None
    if live_bytes == b"":
        live_bytes = None

    result = screen_batch(docs_payload, live_bytes)
    return result.model_dump()


@app.post("/api/face/verify", response_model=FaceVerifyResponse)
async def api_face_verify(
    document: UploadFile = File(...),
    live_photo: UploadFile = File(...),
    document_type: DocumentType | None = Form(None),
):
    document_bytes = await document.read()
    live_bytes = await live_photo.read()
    if not document_bytes or len(document_bytes) > MAX_BYTES:
        raise HTTPException(400, "Document image is missing or larger than 8 MB.")
    if not live_bytes or len(live_bytes) > MAX_BYTES:
        raise HTTPException(400, "Live capture is missing or larger than 8 MB.")
    dtype = document_type.value if document_type else None
    return verify_faces(document_bytes, live_bytes, dtype)


@app.post("/api/demo")
async def api_demo(document_type: DocumentType = Form(DocumentType.PASSPORT)):
    if document_type == DocumentType.NATIONAL_ID:
        blob = aadhaar_specimen()
    elif document_type == DocumentType.VISA:
        blob = visa_specimen()
    elif document_type == DocumentType.DRIVING_LICENSE:
        blob = dl_specimen()
    else:
        blob = passport_specimen()
    result = screen(document_type, blob, None)
    return result.model_dump()


@app.post("/api/demo/batch")
async def api_demo_batch():
    specimens = batch_specimens()
    result = screen_batch(specimens, None)
    return result.model_dump()

