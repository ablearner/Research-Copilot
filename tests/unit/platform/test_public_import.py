import pytest
from services.platform.public_import import arxiv_pdf_url


def test_public_import_origin_and_identifier_validation():
    assert arxiv_pdf_url("2401.12345v2") == "https://arxiv.org/pdf/2401.12345v2"
    assert arxiv_pdf_url("hep-th/9901001") == "https://arxiv.org/pdf/hep-th/9901001"
    for invalid in [
        "http://127.0.0.1/secret",
        "../../etc/passwd",
        "2401.12345?redirect=localhost",
        "2401.12345#fragment",
        "",
    ]:
        with pytest.raises(ValueError):
            arxiv_pdf_url(invalid)
