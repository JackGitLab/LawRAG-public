"""Public snapshot must not reintroduce private document properties or links."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "kb").glob("*.docx")))
def test_corpus_has_no_personal_document_properties(path):
    with zipfile.ZipFile(path) as archive:
        for name in ("docProps/core.xml", "docProps/custom.xml"):
            assert len(ET.fromstring(archive.read(name))) == 0
        app = ET.fromstring(archive.read("docProps/app.xml"))
        for node in app:
            if node.tag.split("}")[-1] in {"Manager", "Company", "Template", "HyperlinkBase"}:
                assert not node.text
        assert ET.fromstring(archive.read("word/document.xml")) is not None


def test_showcase_does_not_link_to_private_repository():
    private_url = re.compile(r"https://github\.com/JackGitLab/LawRAG(?:/|\.git\b)")
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        if path.name != "ERRORS.md":
            assert not private_url.search(path.read_text(encoding="utf-8")), path
