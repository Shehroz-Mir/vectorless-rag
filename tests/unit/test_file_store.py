from pathlib import Path
from uuid import uuid4

import pytest

from vectorless_rag.operations import FileStore, user_key_for
from vectorless_rag.storage import LocalFileStore, safe_filename

ALICE = user_key_for("alice")


def test_original_is_saved_under_the_user_and_document(tmp_path: Path) -> None:
    store: FileStore = LocalFileStore(tmp_path)
    document_id = uuid4()

    path = store.save_original(ALICE, document_id, "Manual.pdf", b"%PDF-1.7 data")

    assert path == tmp_path.resolve() / "users" / ALICE / "documents" / str(document_id) / "original" / "Manual.pdf"
    assert path.read_bytes() == b"%PDF-1.7 data"
    assert not list(path.parent.glob("*.part"))


def test_enriched_copy_keeps_the_name_in_its_own_folder(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    document_id = uuid4()

    original = store.save_original(ALICE, document_id, "Manual.pdf", b"x")
    enriched = store.enriched_path(ALICE, document_id, "Manual.pdf")

    assert enriched.name == original.name and enriched.parent.name == "enriched"
    assert not enriched.exists()  # the enrichment step writes it


def test_delete_removes_only_that_documents_folder(tmp_path: Path) -> None:
    store = LocalFileStore(tmp_path)
    kept, removed = uuid4(), uuid4()
    kept_path = store.save_original(ALICE, kept, "a.pdf", b"x")
    removed_path = store.save_original(ALICE, removed, "b.pdf", b"x")

    store.delete_document_files(ALICE, removed)
    store.delete_document_files(ALICE, removed)  # a second delete is harmless

    assert kept_path.exists() and not removed_path.parent.parent.exists()


def test_user_keys_must_be_real_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a user key"):
        LocalFileStore(tmp_path).save_original("../../etc", uuid4(), "a.pdf", b"x")


@pytest.mark.parametrize("raw, expected", [
    ("Manual.pdf", "Manual.pdf"),
    ("REPORT.PDF", "REPORT.pdf"),
    ("../../evil.pdf", "evil.pdf"),
    ("C:\\Users\\x\\scan.pdf", "scan.pdf"),
    ("folder/sub/notes", "notes.pdf"),
    ("TD I-Series (v1.1).pdf", "TD I-Series (v1.1).pdf"),
    ("bad<>:\"|?*name.pdf", "bad_name.pdf"),
    ("Übersicht.pdf", "Übersicht.pdf"),
    ("CON.pdf", "_CON.pdf"),
    ("...pdf", "document.pdf"),
    ("", "document.pdf"),
])
def test_file_names_are_made_safe(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_long_names_are_shortened_but_keep_the_extension() -> None:
    name = safe_filename("x" * 400 + ".pdf")

    assert len(name) == 150 and name.endswith(".pdf")
