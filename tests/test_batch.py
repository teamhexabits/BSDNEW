from app.demo_specimen import (
    aadhaar_specimen,
    dl_specimen,
    passport_specimen,
    visa_specimen,
    passport_ocr_text,
    visa_ocr_text,
    aadhaar_ocr_text,
    dl_ocr_text,
    batch_specimens,
)
from app.documents.patterns import dl_number_ok, is_masked_aadhaar, name_similarity, verhoeff_ok
from app.modules.ocr import detect_document_type, extract_fields
from app.modules.validation import cross_validate, validate
from app.pipeline import screen_batch
from app.schemas import DocumentType, RiskLevel


def test_name_similarity():
    assert name_similarity("RAJESH KUMAR", "KUMAR RAJESH") == 1.0
    assert name_similarity("SONAVANE PRAKASH SHAMARAO", "PRAKASH SHAMARAO SONAVANE") == 1.0
    assert name_similarity("SONAVANE PRAKASH", "PRAKASH SONAVANE") == 1.0
    assert name_similarity("SONAVANE", "PRAKASH SHAMARAO SONAVANE") >= 0.85
    assert name_similarity("SONAVANE", "SONAVANE PRAKASH SHAMARAO") >= 0.85
    assert name_similarity("PRAKASH", "SONAVANE PRAKASH SHAMARAO") >= 0.85
    assert name_similarity("PRAKASH", "PRAKASH SHAMARAO SONAVANE") >= 0.85
    assert name_similarity("PRAKASH SONAVANE", "SONAVANE PRAKASH SHAMARAO") >= 0.90
    assert name_similarity("PRAKASH S SONAVANE", "SONAVANE PRAKASH SHAMARAO") >= 0.90
    assert name_similarity("MR. RAJESH KUMAR", "RAJESH KUMAR") == 1.0
    assert name_similarity("RAJESH KUMAR SHARMA", "RAJESH KUMAR") >= 0.85
    assert name_similarity("RAJESH KUMAR", "SURESH PATEL") <= 0.50
    assert name_similarity("RAJESH KUMAR", "SURESH KUMAR") <= 0.50


def test_name_order_and_partial_cross_document_matching():
    docs = [
        {
            "filename": "passport.png",
            "doc_type": "indian_passport",
            "fields": {
                "passport_number": "F4608240",
                "surname": "SONAVANE",
                "given_names": "PRAKASH SHAMARAO",
                "full_name": "PRAKASH SHAMARAO SONAVANE",  # Given names then surname
                "date_of_birth": "16/08/1951",
            },
        },
        {
            "filename": "visa.png",
            "doc_type": "visa",
            "fields": {
                "visa_number": "0415765",
                "passport_number": "F4608240",
                "full_name": "SONAVANE PRAKASH SHAMARAO",  # Surname then given names
                "date_of_birth": "16/08/1951",
            },
        },
        {
            "filename": "aadhaar.png",
            "doc_type": "national_id",
            "fields": {
                "aadhaar_number": "499118665241",
                "full_name": "SONAVANE",  # Only surname written
                "date_of_birth": "16/08/1951",
            },
        },
        {
            "filename": "dl.png",
            "doc_type": "driving_license",
            "fields": {
                "dl_number": "MH0120180012345",
                "full_name": "PRAKASH",  # Only first name written
                "date_of_birth": "16/08/1951",
            },
        },
    ]
    checks, findings = cross_validate(docs)
    check_map = {c.name: c.status for c in checks}
    assert check_map.get("Name Consistency") == "MATCH"
    assert check_map.get("Date of Birth Consistency") == "MATCH"
    assert check_map.get("Visa-to-Passport Linkage") == "MATCH"


def test_masked_aadhaar_validation():
    masked = "XXXX XXXX 6619"
    assert is_masked_aadhaar(masked)
    assert verhoeff_ok(masked) is True

    fields = {
        "aadhaar_number": "XXXX XXXX 6619",
        "masked": True,
        "full_name": "PRAKASH RANJAN",
        "date_of_birth": "05/07/1994",
        "sex": "M",
    }
    result, findings = validate(DocumentType.NATIONAL_ID, fields)
    assert result.details["checks"].get("aadhaar_verhoeff") is True
    assert not any(f.code == "AADHAAR_CHECKSUM" for f in findings)


def test_flexible_dl_formats():
    assert dl_number_ok("MH12 20260012345")
    assert dl_number_ok("DL-1420110012345")
    assert dl_number_ok("KA01/2019/0001234")
    assert dl_number_ok("TN-01-20200001234")
    assert not dl_number_ok("XX1234567890123")  # Invalid state code XX


def test_document_auto_detection():
    p_text = "REPUBLIC OF INDIA\nPASSPORT\nType P\nP<INDRAJESH<<KUMAR<<<<<<<<<<<<<<<<<<<\nA1234567<4IND9203120M3203120<<<<<<<<<<<<<<<6"
    assert detect_document_type(p_text) == DocumentType.PASSPORT

    v_text = "REPUBLIC OF INDIA VISA SPECIMEN\nVisa number IN12345678\nTOURIST\nValid from 01/01/2026\n"
    assert detect_document_type(v_text) == DocumentType.VISA

    a_text = "GOVERNMENT OF INDIA\nAADHAAR SPECIMEN\nName: TEST HOLDER\nXXXX XXXX 1234\n"
    assert detect_document_type(a_text) == DocumentType.NATIONAL_ID

    d_text = "GOVERNMENT OF INDIA\nDRIVING LICENCE\nDL No MH-01-2018-0012345\n"
    assert detect_document_type(d_text) == DocumentType.DRIVING_LICENSE


def test_cross_document_matching_dossier():
    docs = [
        {
            "filename": "passport.png",
            "doc_type": "indian_passport",
            "fields": {
                "passport_number": "A1234567",
                "full_name": "RAJESH KUMAR",
                "date_of_birth": "12/03/1992",
                "sex": "M",
                "date_of_expiry": "12/03/2032",
            },
        },
        {
            "filename": "visa.png",
            "doc_type": "visa",
            "fields": {
                "visa_number": "IN12345678",
                "passport_number": "A1234567",
                "full_name": "RAJESH KUMAR",
                "valid_until": "01/01/2027",
            },
        },
        {
            "filename": "aadhaar.png",
            "doc_type": "national_id",
            "fields": {
                "aadhaar_number": "499118665241",
                "full_name": "KUMAR RAJESH",
                "date_of_birth": "12/03/1992",
                "sex": "M",
            },
        },
    ]
    checks, findings = cross_validate(docs)
    check_map = {c.name: c.status for c in checks}
    assert check_map.get("Name Consistency") == "MATCH"
    assert check_map.get("Date of Birth Consistency") == "MATCH"
    assert check_map.get("Gender Consistency") == "MATCH"
    assert check_map.get("Visa-to-Passport Linkage") == "MATCH"
    assert not any(f.severity == "high" for f in findings)


def test_cross_document_mismatch_detected():
    docs = [
        {
            "filename": "passport.png",
            "doc_type": "indian_passport",
            "fields": {
                "passport_number": "A1234567",
                "full_name": "RAJESH KUMAR",
                "date_of_birth": "12/03/1992",
            },
        },
        {
            "filename": "visa.png",
            "doc_type": "visa",
            "fields": {
                "visa_number": "IN12345678",
                "passport_number": "Z9999999",  # Mismatching passport number
                "full_name": "SURESH PATEL",    # Mismatching name
                "date_of_birth": "01/01/1980",  # Mismatching DOB
            },
        },
    ]
    checks, findings = cross_validate(docs)
    check_map = {c.name: c.status for c in checks}
    assert check_map.get("Name Consistency") == "MISMATCH"
    assert check_map.get("Date of Birth Consistency") == "MISMATCH"
    assert check_map.get("Visa-to-Passport Linkage") == "MISMATCH"
    high_codes = {f.code for f in findings if f.severity == "high"}
    assert "CROSS_DOC_NAME_MISMATCH" in high_codes
    assert "CROSS_DOC_DOB_MISMATCH" in high_codes
    assert "VISA_PASSPORT_MISMATCH" in high_codes


def test_end_to_end_batch_specimens():
    specimens = batch_specimens()
    result = screen_batch(specimens, None)
    assert result.batch_id.startswith("BATCH-")
    assert len(result.documents) == 4
    assert result.dossier.primary_name is not None
    assert result.dossier.passport_number == "A1234567"
    assert result.dossier.visa_number == "IN12345678"
    assert result.overall_authenticity_score >= 70
