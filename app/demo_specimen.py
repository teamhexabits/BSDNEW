"""Runtime SPECIMEN images for demos — clearly labelled, not real credentials."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from app.documents.patterns import mrz_compute, verhoeff_check_digit


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def passport_mrz() -> tuple[str, str]:
    number9 = "A1234567<"
    cd_num = mrz_compute(number9)
    dob = "920312"
    cd_dob = mrz_compute(dob)
    expiry = "320312"
    cd_exp = mrz_compute(expiry)
    personal = "<" * 14
    cd_opt = mrz_compute(personal)
    composite = mrz_compute(number9 + cd_num + dob + cd_dob + expiry + cd_exp + personal + cd_opt)
    mrz1 = "P<INDSPECIMEN<<TRAVELLER<TEST<<<<<<<<<<<<<<<"
    mrz2 = f"{number9}{cd_num}IND{dob}{cd_dob}M{expiry}{cd_exp}{personal}{cd_opt}{composite}"
    return mrz1, mrz2


def specimen_aadhaar() -> str:
    body = "49911866524"
    return body + verhoeff_check_digit(body)


def passport_ocr_text() -> str:
    mrz1, mrz2 = passport_mrz()
    return (
        "REPUBLIC OF INDIA SPECIMEN\nPASSPORT P IND\n"
        "Surname SPECIMEN\nGiven names TRAVELLER TEST\n"
        "Passport No A1234567\nNationality IND\n"
        "Date of birth 12/03/1992\nSex M\nExpiry 12/03/2032\n"
        f"{mrz1}\n{mrz2}\n"
    )


def aadhaar_ocr_text() -> str:
    grouped = specimen_aadhaar()
    grouped = f"{grouped[:4]} {grouped[4:8]} {grouped[8:]}"
    return (
        "GOVERNMENT OF INDIA AADHAAR SPECIMEN\n"
        f"Name: TRAVELLER TEST\nDOB: 12/03/1992\nGender: M\n{grouped}\n"
    )


def visa_ocr_text() -> str:
    return (
        "REPUBLIC OF INDIA VISA SPECIMEN\n"
        "Visa number IN12345678\nTOURIST\n"
        "Valid from 01/01/2026\nValid until 01/01/2027\nPassport No A1234567\n"
    )


def dl_ocr_text() -> str:
    return (
        "GOVERNMENT OF INDIA DRIVING LICENCE SPECIMEN\n"
        "DL number MH-01-2018-0012345\nName TRAVELLER TEST\nDate of birth 12/03/1992\n"
        "Valid until 12/03/2032\n"
    )


def passport_specimen() -> bytes:
    image = Image.new("RGB", (1012, 638), (12, 48, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 988, 614), outline=(201, 162, 39), width=4)
    draw.text((48, 40), "REPUBLIC OF INDIA  •  SPECIMEN", fill=(201, 162, 39), font=_font(28))
    draw.text((48, 90), "PASSPORT  P  IND", fill=(240, 240, 230), font=_font(22))
    draw.rectangle((48, 140, 280, 420), fill=(210, 190, 160))
    draw.text((70, 240), "PHOTO", fill=(80, 60, 40), font=_font(22))
    draw.text((310, 150), "Surname  SPECIMEN", fill="white", font=_font(22))
    draw.text((310, 190), "Given names  TRAVELLER TEST", fill="white", font=_font(22))
    draw.text((310, 240), "Passport No  A1234567", fill="white", font=_font(22))
    draw.text((310, 280), "Nationality  IND", fill="white", font=_font(22))
    draw.text((310, 320), "Date of birth  12/03/1992", fill="white", font=_font(22))
    draw.text((310, 360), "Sex  M     Expiry  12/03/2032", fill="white", font=_font(22))
    mrz1, mrz2 = passport_mrz()
    draw.rectangle((40, 480, 972, 600), fill=(230, 230, 220))
    draw.text((50, 500), mrz1, fill="black", font=_font(18))
    draw.text((50, 540), mrz2, fill="black", font=_font(18))
    draw.text((700, 40), "NOT A REAL DOCUMENT", fill=(220, 80, 70), font=_font(18))
    return _png(image)


def visa_specimen() -> bytes:
    image = Image.new("RGB", (1012, 638), (18, 36, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 988, 614), outline=(201, 162, 39), width=4)
    draw.text((48, 40), "REPUBLIC OF INDIA  •  VISA SPECIMEN", fill=(201, 162, 39), font=_font(26))
    draw.text((48, 110), "Visa number  IN12345678", fill="white", font=_font(24))
    draw.text((48, 160), "TOURIST", fill="white", font=_font(22))
    draw.text((48, 210), "Valid from  01/01/2026", fill="white", font=_font(22))
    draw.text((48, 260), "Valid until  01/01/2027", fill="white", font=_font(22))
    draw.text((48, 310), "Passport No  A1234567", fill="white", font=_font(22))
    draw.text((48, 540), "SPECIMEN ONLY — NOT A REAL VISA", fill=(220, 80, 70), font=_font(18))
    return _png(image)


def dl_specimen() -> bytes:
    image = Image.new("RGB", (1012, 638), (248, 248, 242))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1012, 70), fill=(18, 80, 40))
    draw.text((24, 18), "GOVERNMENT OF INDIA  •  DRIVING LICENCE SPECIMEN", fill="white", font=_font(22))
    draw.text((48, 110), "DL number  MH-01-2018-0012345", fill=(20, 20, 20), font=_font(24))
    draw.text((48, 170), "Name  TRAVELLER TEST", fill=(20, 20, 20), font=_font(22))
    draw.text((48, 220), "Date of birth  12/03/1992", fill=(20, 20, 20), font=_font(22))
    draw.text((48, 270), "Valid until  12/03/2032", fill=(20, 20, 20), font=_font(22))
    draw.text((48, 540), "SPECIMEN ONLY — NOT AN OFFICIAL LICENCE", fill=(160, 40, 40), font=_font(18))
    return _png(image)


def aadhaar_specimen() -> bytes:
    image = Image.new("RGB", (900, 560), (245, 248, 252))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 900, 70), fill=(18, 60, 140))
    draw.text((24, 18), "GOVERNMENT OF INDIA  •  AADHAAR SPECIMEN", fill="white", font=_font(22))
    draw.rectangle((40, 100, 250, 360), fill=(200, 200, 200))
    draw.text((90, 210), "PHOTO", fill=(80, 80, 80), font=_font(20))
    draw.text((280, 120), "Name: TRAVELLER TEST", fill=(20, 20, 20), font=_font(22))
    draw.text((280, 170), "DOB: 12/03/1992", fill=(20, 20, 20), font=_font(22))
    draw.text((280, 220), "Gender: M", fill=(20, 20, 20), font=_font(22))
    aadhaar = specimen_aadhaar()
    grouped = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
    draw.text((280, 300), grouped, fill=(18, 60, 140), font=_font(32))
    draw.text((40, 500), "SPECIMEN ONLY — NOT AN OFFICIAL ID", fill=(160, 40, 40), font=_font(18))
    return _png(image)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def batch_specimens() -> list[dict[str, Any]]:
    """Return a 4-document coordinated dossier for a single traveller."""
    return [
        {
            "document_bytes": passport_specimen(),
            "document_type": "indian_passport",
            "filename": "passport_specimen.png",
            "text_hint": passport_ocr_text(),
        },
        {
            "document_bytes": visa_specimen(),
            "document_type": "visa",
            "filename": "visa_specimen.png",
            "text_hint": visa_ocr_text(),
        },
        {
            "document_bytes": aadhaar_specimen(),
            "document_type": "national_id",
            "filename": "aadhaar_specimen.png",
            "text_hint": aadhaar_ocr_text(),
        },
        {
            "document_bytes": dl_specimen(),
            "document_type": "driving_license",
            "filename": "dl_specimen.png",
            "text_hint": dl_ocr_text(),
        },
    ]

