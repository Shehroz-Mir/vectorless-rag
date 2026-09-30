"""Guards for the package-interface style (CLAUDE.md): each package's __init__.py exposes its public
names; the shared layers stay light; the top level re-exports nothing."""
import importlib
import subprocess
import sys

import pytest

import vectorless_rag

PACKAGES = [
    "vectorless_rag.models", "vectorless_rag.operations", "vectorless_rag.pdf", "vectorless_rag.db",
    "vectorless_rag.storage", "vectorless_rag.vision", "vectorless_rag.worker", "vectorless_rag.indexing",
]
ADAPTER_LIBRARIES = ["pymupdf", "langchain", "langchain_openai", "pageindex", "litellm", "openai", "sqlalchemy"]


@pytest.mark.parametrize("package", PACKAGES)
def test_every_listed_public_name_exists(package: str) -> None:
    module = importlib.import_module(package)

    assert module.__all__, f"{package} exposes nothing"
    assert [name for name in module.__all__ if not hasattr(module, name)] == []


def test_models_and_operations_load_no_adapter_library() -> None:
    """A fresh interpreter, so libraries other tests already imported do not count."""
    probe = (
        "import sys, vectorless_rag.models, vectorless_rag.operations; "
        f"print([m for m in {ADAPTER_LIBRARIES!r} if m in sys.modules])"
    )

    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()

    assert loaded == "[]"


def test_top_level_package_re_exports_nothing() -> None:
    """Fresh interpreter: once imported, subpackages appear as attributes of their parent anyway."""
    probe = "import vectorless_rag; print([n for n in vars(vectorless_rag) if not n.startswith('_')])"

    public = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()

    assert public == "[]"
    assert vectorless_rag.__doc__
