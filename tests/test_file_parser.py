from __future__ import annotations

import pytest

from app.file_parser import FileParser, UnsupportedFileTypeError


def test_parse_plain_text_file() -> None:
    parser = FileParser()
    content = parser.parse_bytes("tema.txt", b"Dette er en test")
    assert "test" in content


def test_parse_bytes_unsupported_extension_raises() -> None:
    parser = FileParser()
    with pytest.raises(UnsupportedFileTypeError):
        parser.parse_bytes("tema.xls", b"data")
