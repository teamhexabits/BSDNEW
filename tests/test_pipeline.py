from app.documents.patterns import (
    dl_number_ok,
    format_dl_number,
    mrz_compute,
    passport_number_ok,
    verhoeff_check_digit,
    verhoeff_ok,
)
from app.demo_specimen import passport_specimen, aadhaar_specimen
from app.pipeline import screen
from app.schemas import DocumentType


def test_passport_number_formats():
    assert passport_number_ok("A1234567")
    assert passport_number_ok("AB123456")
    assert passport_number_ok("ab-123456")
    assert not passport_number_ok("A123456")
    assert not passport_number_ok("ABC12345")
    assert not passport_number_ok("A12345678")
    assert not passport_number_ok("12345678")


def test_verhoeff_roundtrip():
    body = "49911866524"
    number = body + verhoeff_check_digit(body)
    assert len(number) == 12
    assert verhoeff_ok(number)
    assert not verhoeff_ok(number[:-1] + ("0" if number[-1] != "0" else "1"))


def test_mrz_line_length():
    from app.demo_specimen import passport_mrz

    mrz1, mrz2 = passport_mrz()
    assert len(mrz1) == 44
    assert len(mrz2) == 44


def test_specimen_passport_pipeline():
    from app.demo_specimen import passport_ocr_text

    result = screen(DocumentType.PASSPORT, passport_specimen(), text_hint=passport_ocr_text())
    assert result.extracted_fields.get("passport_number") == "A1234567"
    assert 0 <= result.authenticity_score <= 100


def test_two_letter_passport_extraction():
    from app.modules.ocr import extract_fields
    from app.modules.validation import validate

    text = "REPUBLIC OF INDIA\nPassport No AB123456\nDate of birth 12/03/1992\nExpiry 12/03/2032\n"
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields.get("passport_number") == "AB123456"
    result, findings = validate(DocumentType.PASSPORT, fields)
    assert result.details["checks"].get("passport_format") is True
    assert not any(f.code == "PASSPORT_FORMAT" for f in findings)


def test_passport_visual_zone_and_spaced_mrz():
    from app.modules.ocr import extract_fields

    text = (
        "Surname  KUMAR\nGiven names  RAJESH\nPassport No. A1092837\n"
        "Nationality INDIAN\nDate of Birth 12/03/1992\nDate of Issue 01/01/2020\n"
        "Date of Expiry 31/12/2029\nSex M\nPlace of Birth DELHI\nPlace of Issue MUMBAI\n"
        "P < I N D K U M A R < < R A J E S H < < < < < < < < < < < < < < < < < < < < < < < <\n"
        "A 1 0 9 2 8 3 7 < 0 I N D 9 2 0 3 1 2 0 M 2 9 1 2 3 1 0 < < < < < < < < < < < < < < 0\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["passport_number"] == "A1092837"
    assert fields["surname"] == "KUMAR"
    assert "RAJESH" in fields["given_names"]
    assert fields["date_of_birth"] == "12/03/1992"
    assert fields["date_of_issue"] == "01/01/2020"
    assert fields["date_of_expiry"] == "31/12/2029"
    assert fields["place_of_birth"] == "DELHI"
    assert fields["mrz_present"] is True


def test_visual_zone_without_mrz_keeps_names():
    from app.modules.ocr import extract_fields

    text = (
        "Surname KUMAR\nGiven names RAJESH\nPassport No. AB123456\n"
        "Date of Birth 12 Mar 1992\nDate of Issue 01/01/2020\n"
        "Expiry 31/12/2029\nSex M\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["passport_number"] == "AB123456"
    assert fields["surname"] == "KUMAR"
    assert fields["given_names"] == "RAJESH"
    assert fields.get("mrz_present") is None
    assert "PASSPORTNO" not in str(fields.get("surname"))


def test_visa_number_ignores_words():
    from app.modules.ocr import extract_fields

    text = "REPUBLIC OF INDIA VISA SPECIMEN\nVisa number IN12345678\nTOURIST\n"
    fields = extract_fields(DocumentType.VISA, text)
    assert fields.get("visa_number") == "IN12345678"


def test_indian_passport_sample_layout():
    from app.modules.ocr import extract_fields

    text = (
        "REPUBLIC OF INDIA\nType / P\nCountry Code / IND\nPassport No. / R2065510\n"
        "Surname /\nGiven Name(s) /\nASMA\nNationality / INDIAN\nSex / F\n"
        "Date of Birth / 30/07/1987\nPlace of Birth / HYDERABAD, TELANGANA\n"
        "Place of Issue / HYDERABAD\nDate of Issue / 05/09/2017\n"
        "Date of Expiry / 04/09/2027\n"
        "P<IND<<ASMA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "R2065510<4IND8707305F2709042<<<<<<<<<<<<<<<6\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["passport_number"] == "R2065510"
    assert fields["given_names"] == "ASMA"
    assert fields.get("surname") in (None, "")
    assert fields["sex"] == "F"
    assert fields["nationality"] == "IND"
    assert fields["date_of_birth"] == "30/07/1987"
    assert fields["date_of_issue"] == "05/09/2017"
    assert fields["date_of_expiry"] == "04/09/2027"
    assert "HYDERABAD" in fields["place_of_birth"]
    assert fields["mrz_present"] is True


def test_noisy_passport_mrz_repairs_dates():
    from app.modules.ocr import extract_fields

    text = (
        "Passport No.\nR2065510\nGiven Name(s)\nASMA\nVIRART/INDIAN F\n"
        "HYDERABAD, TELANGANA\nHYDERABAD\n05/09/2017 04/09/2027\n"
        "R2065510<41 ND8707305 F2 709042 <<<<ccceecceeccg\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["passport_number"] == "R2065510"
    assert fields["given_names"] == "ASMA"
    assert fields["date_of_birth"] == "30/07/1987"
    assert fields["date_of_expiry"] == "04/09/2027"
    assert fields["sex"] == "F"


def test_issue_expiry_pair_without_expiry_label():
    from app.modules.ocr import extract_fields

    text = (
        "Surname KUMAR\nGiven names RAJESH\nPassport No. A1092837\n"
        "Date of Birth 12/03/1992\nDate of Issue / Date of Expiry\n01/01/2020 31/12/2029\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["date_of_issue"] == "01/01/2020"
    assert fields["date_of_expiry"] == "31/12/2029"


def test_mrz_ocr_confusions_pass_validation():
    from app.documents.patterns import mrz_compute, td3_line2_checks
    from app.modules.ocr import extract_fields
    from app.modules.validation import validate

    number = "R2065510<"
    dob = "870730"
    expiry = "270904"
    optional = "<" * 14
    line2 = (
        f"{number}{mrz_compute(number)}IND{dob}{mrz_compute(dob)}F"
        f"{expiry}{mrz_compute(expiry)}{optional}{mrz_compute(optional)}"
    )
    composite = mrz_compute(line2[0:10] + line2[13:20] + line2[21:43])
    line2 = line2 + composite
    assert len(line2) == 44
    noisy = (
        "Passport No. R2065510\nDate of Birth 30/07/1987\nDate of Expiry 04/09/2027\n"
        "P<IND<<ASMA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        + line2.replace("<", "C", 6).replace("0", "O", 1)
    )
    fields = extract_fields(DocumentType.PASSPORT, noisy)
    assert fields["date_of_expiry"] == "04/09/2027"
    assert fields["passport_number"] == "R2065510"
    checks = td3_line2_checks(fields["mrz_line2"])
    assert checks.get("mrz_passport_cd") is True
    assert checks.get("mrz_expiry_cd") is True
    result, findings = validate(DocumentType.PASSPORT, fields)
    assert result.details["checks"].get("mrz_passport_cd") is True
    assert result.details["checks"].get("mrz_expiry_cd") is True
    assert not any(f.code == "MRZ_MULTIPLE_CHECK_FAIL" for f in findings)


def test_visual_expiry_wins_over_broken_mrz_date():
    from app.modules.ocr import extract_fields

    text = (
        "Passport No. R2065510\nDate of Birth 30/07/1987\nDate of Issue 05/09/2017\n"
        "Date of Expiry 04/09/2027\n"
        "P<IND<<ASMA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "R2065510<4IND8707305Fxxxxxx2<<<<<<<<<<<<<<<6\n"
    )
    fields = extract_fields(DocumentType.PASSPORT, text)
    assert fields["date_of_expiry"] == "04/09/2027"


def test_aadhaar_sample_layout():
    from app.modules.ocr import extract_fields

    text = (
        "GOVERNMENT OF INDIA\nPrakash Ranjan\n"
        "जन्म तिथि / DOB: 05/07/1994\nपुरुष / MALE\n9183 0074 6619\n"
    )
    fields = extract_fields(DocumentType.NATIONAL_ID, text)
    assert fields["aadhaar_number"] == "918300746619"
    assert fields["full_name"] == "PRAKASH RANJAN"
    assert fields["date_of_birth"] == "05/07/1994"
    assert fields["sex"] == "M"


def test_maharashtra_dl_sample_layout():
    from app.modules.ocr import extract_fields
    from app.modules.validation import validate

    assert dl_number_ok("MH12 20260012345")
    assert format_dl_number("MH1220260012345") == "MH12 20260012345"
    text = (
        "THE UNION OF INDIA\nMAHARASHTRA STATE MOTOR DRIVING LICENCE\n"
        "DL No: MH12 20260012345 DOI: 15-08-2024\n"
        "Valid Till: 14-08-2044 (NT)\n"
        "MCWG 15-08-2024\nLMV 15-08-2024\n"
        "DOB: 15-08-1994 Blood Group: B+\n"
        "Name: RAHUL KUMAR\nS/D/W of: SURESH KUMAR\n"
        "Address: 42 MODEL COLONY, PUNE, MAHARASHTRA 411016\n"
    )
    fields = extract_fields(DocumentType.DRIVING_LICENSE, text)
    assert fields["dl_number"] == "MH12 20260012345"
    assert fields["issuing_state"] == "MH"
    assert fields["full_name"] == "RAHUL KUMAR"
    assert fields["date_of_birth"] == "15/08/1994"
    assert fields["date_of_issue"] == "15/08/2024"
    assert fields["date_of_expiry"] == "14/08/2044"
    assert fields["blood_group"] == "B+"
    assert fields["parent_or_spouse"] == "SURESH KUMAR"
    result, findings = validate(DocumentType.DRIVING_LICENSE, fields)
    assert result.details["checks"].get("dl_format") is True
    assert not any(f.code == "DL_FORMAT" for f in findings)


def test_specimen_aadhaar_pipeline():
    from app.demo_specimen import aadhaar_ocr_text

    result = screen(DocumentType.NATIONAL_ID, aadhaar_specimen(), text_hint=aadhaar_ocr_text())
    assert result.validation.details["checks"].get("aadhaar_verhoeff") is True
    assert 0 <= result.authenticity_score <= 100


def test_sample_passport_photo_extraction():
    from pathlib import Path
    sample = Path(r"C:\Users\Neel\.gemini\antigravity-ide\brain\64a20a37-7d1b-4898-ae3a-c9a7a2f076c5\.user_uploaded\media_1789757752855.jpg")
    if not sample.is_file():
        return
    result = screen(DocumentType.PASSPORT, sample.read_bytes())
    fields = result.extracted_fields
    assert fields["passport_number"] == "F4608240"
    assert fields["mrz_number"] == "F4608240"
    assert fields["surname"] == "SONAVANE"
    assert fields["given_names"] == "PRAKASH SHAMARAO"
    assert fields["date_of_birth"] == "16/08/1951"
    assert fields["place_of_birth"] == "KUMATHE SATARA"
    assert fields["place_of_issue"] == "PUNE"
    assert fields["date_of_issue"] == "12/09/2005"
    assert fields["date_of_expiry"] == "11/09/2015"
    assert fields["mrz_line1"] == "P<INDSONAVANE<<PRAKASH<SHAMARAO<<<<<<<<<<<<<"
    assert fields["mrz_line2"] == "F4608240<7IND5108163M1509119<<<<<<<<<<<<<<<2"
    checks = result.validation.details.get("checks", {})
    assert checks.get("mrz_passport_cd") is True
    assert checks.get("mrz_dob_cd") is True
    assert checks.get("mrz_expiry_cd") is True
    assert checks.get("mrz_composite") is True
    assert not any(f.code == "MRZ_MULTIPLE_CHECK_FAIL" for f in result.findings)


def test_sample_visa_photo_extraction():
    from pathlib import Path
    from app.modules.ocr import classify_document

    sample = Path(
        r"C:\Users\Neel\.gemini\antigravity-ide\brain\64a20a37-7d1b-4898-ae3a-c9a7a2f076c5\.user_uploaded\media_1789817102427.jpg"
    )
    if not sample.is_file():
        return
    data = sample.read_bytes()
    doc_type, _, _ = classify_document(data)
    assert doc_type == DocumentType.VISA

    result = screen(DocumentType.VISA, data)
    fields = result.extracted_fields
    assert "SONAVANE" in str(fields.get("full_name"))
    assert "PRAKASH" in str(fields.get("full_name"))
    assert fields.get("surname") == "SONAVANE"
    assert fields.get("given_names") == "PRAKASH SHAMARAO"
    assert fields.get("date_of_birth") == "16/08/1951"
    assert fields.get("sex") == "M"
    assert fields.get("nationality") == "IND"
    assert fields.get("passport_number") == "F4608240"
    assert fields.get("valid_from") == "16/09/2005"
    assert fields.get("valid_until") == "21/10/2005"
    assert fields.get("visa_type") == "MULTIPLE JOURNEY"
    assert fields.get("period_of_stay") == "SHORT VISIT"
    assert fields.get("remarks") == "Not Valid for Employment"
    assert "SINGAPORE" in str(fields.get("issuing_country"))
    assert result.validation.details["checks"].get("visa_present") is True
    assert result.validation.details["checks"].get("visa_window") is True
    assert result.validation.details["checks"].get("passport_link_present") is True


def test_simultaneous_passport_and_visa_screening():
    from pathlib import Path
    from app.pipeline import screen_batch

    p_sample = Path(
        r"C:\Users\Neel\.gemini\antigravity-ide\brain\64a20a37-7d1b-4898-ae3a-c9a7a2f076c5\.user_uploaded\media_1789757752855.jpg"
    )
    v_sample = Path(
        r"C:\Users\Neel\.gemini\antigravity-ide\brain\64a20a37-7d1b-4898-ae3a-c9a7a2f076c5\.user_uploaded\media_1789817102427.jpg"
    )
    if not p_sample.is_file() or not v_sample.is_file():
        return

    batch_input = [
        {"filename": "passport.jpg", "document_bytes": p_sample.read_bytes(), "document_type": None},
        {"filename": "visa.jpg", "document_bytes": v_sample.read_bytes(), "document_type": None},
    ]
    batch_result = screen_batch(batch_input)
    assert len(batch_result.documents) == 2
    types = {d.document_type for d in batch_result.documents}
    assert DocumentType.PASSPORT in types
    assert DocumentType.VISA in types

    check_map = {c.name: c.status for c in batch_result.cross_checks}
    assert check_map.get("Visa-to-Passport Linkage") == "MATCH"
    assert check_map.get("Name Consistency") == "MATCH"
    assert check_map.get("Date of Birth Consistency") == "MATCH"
    assert batch_result.dossier.passport_number == "F4608240"
    assert "SONAVANE" in batch_result.dossier.primary_name

