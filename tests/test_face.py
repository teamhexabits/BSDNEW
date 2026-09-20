from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.modules.face import FACE_MATCH_THRESHOLD, run_face, verify_faces


def _face(area: float, embedding: np.ndarray, det_score: float = 0.9):
    side = area ** 0.5
    return SimpleNamespace(
        bbox=np.array([0.0, 0.0, side, side]),
        det_score=det_score,
        normed_embedding=embedding,
        embedding=embedding,
    )


def test_verify_match_uses_cosine_dot_product():
    vec = np.ones(4) / 2.0
    document = _face(100.0, vec)
    live = _face(80.0, vec)

    with patch("app.modules.face.get_face_app") as mock_app:
        mock_app.return_value.get.side_effect = [[document], [live]]
        with patch("app.modules.face.bytes_to_bgr", side_effect=[np.zeros((8, 8, 3)), np.zeros((8, 8, 3))]):
            result = verify_faces(b"doc", b"live")

    assert result["matched"] is True
    assert result["message"] == "FACE MATCH"
    assert result["similarity"] == 1.0
    assert result["threshold"] == FACE_MATCH_THRESHOLD


def test_live_requires_single_face():
    vec = np.ones(4) / 2.0
    document = _face(100.0, vec)
    live_a = _face(80.0, vec)
    live_b = _face(70.0, vec)

    with patch("app.modules.face.get_face_app") as mock_app:
        mock_app.return_value.get.side_effect = [[document], [live_a, live_b]]
        with patch("app.modules.face.bytes_to_bgr", side_effect=[np.zeros((8, 8, 3)), np.zeros((8, 8, 3))]):
            result = verify_faces(b"doc", b"live")

    assert result["matched"] is False
    assert result["similarity"] is None
    assert result["reason"] == "Multiple faces detected"
    assert result["message"] == "FACE VERIFICATION FAILED"


def test_pipeline_skips_face_without_live_photo():
    module, findings = run_face(b"doc", None)
    assert module.details["reason"] == "missing_live_photo"
    assert any(f.code == "NO_LIVE_PHOTO" for f in findings)
