"""The PageIndex adapter with the real SDK on an empty local library: everything that needs no LLM call.

Indexing itself calls the index model, so it is covered by the opt-in live tests.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vectorless_rag.indexing import PageIndexClientPool
from vectorless_rag.models import IndexCitation
from vectorless_rag.operations import DocumentNotFound, UserIndex, UserIndexProvider, user_key_for

ALICE = user_key_for("alice")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def pool(tmp_path: Path) -> PageIndexClientPool:
    return PageIndexClientPool(tmp_path / "var", api_key="sk-test", model="gpt-5.6-luna", summary_concurrency=2)


def test_each_user_gets_one_cached_index_in_their_own_folder(pool: PageIndexClientPool, tmp_path: Path) -> None:
    bob = user_key_for("bob")

    assert pool.for_user(ALICE) is pool.for_user(ALICE)
    assert pool.for_user(ALICE) is not pool.for_user(bob)
    assert pool.storage_path(ALICE) == (tmp_path / "var").resolve() / "users" / ALICE / "pageindex"


def test_folder_names_come_only_from_user_keys(pool: PageIndexClientPool) -> None:
    with pytest.raises(ValueError, match="not a user key"):
        pool.for_user("../alice")


def test_the_agent_gets_only_the_read_only_tools(pool: PageIndexClientPool) -> None:
    names = sorted(tool.__name__ for tool in pool.for_user(ALICE).agent_tools())

    assert names == ["browse_documents", "get_document", "get_document_structure", "get_page_content"]


def test_instructions_carry_the_tool_guidance_and_the_citation_rules(pool: PageIndexClientPool) -> None:
    instructions = pool.for_user(ALICE).agent_instructions()

    assert "get_page_content" in instructions and "<cite" in instructions


def test_citation_tags_become_plain_numbers(pool: PageIndexClientPool) -> None:
    resolved = pool.for_user(ALICE).resolve_citations(
        'Hot: 60 °C <cite doc="manual.pdf" page="27"/>, again <cite doc="manual.pdf" page="27"/>, '
        'see <cite doc="other.pdf" page="3-4"/>.'
    )

    assert resolved.text == "Hot: 60 °C [1], again [1], see [2]."
    # doc_id None: neither name is in this user's (empty) library, so the query step drops them.
    assert resolved.citations == (
        IndexCitation(index=1, document="manual.pdf", doc_id=None, page=27),
        IndexCitation(index=2, document="other.pdf", doc_id=None, page=3),
    )


def test_unknown_documents(pool: PageIndexClientPool) -> None:
    index = pool.for_user(ALICE)

    with pytest.raises(DocumentNotFound):
        index.document_context(["pi-missing"])
    index.delete("pi-missing")  # already gone is not an error, so a retried delete is safe


def test_adapters_satisfy_the_ports(pool: PageIndexClientPool) -> None:
    provider: UserIndexProvider = pool
    index: UserIndex = pool.for_user(ALICE)

    assert provider is not None and index is not None


PROBE = f"""
import os, pathlib, tempfile
import vectorless_rag.indexing as indexing
pool = indexing.PageIndexClientPool(pathlib.Path(tempfile.mkdtemp()), api_key="sk-test", model="m", summary_concurrency=1)
pool.for_user("{ALICE}")  # a client: PageIndex loads its utils module
import litellm, pageindex.utils
print("OPENAI_API_KEY" in os.environ)
"""


def test_pageindex_and_litellm_do_not_copy_the_env_file_into_the_environment() -> None:
    """Both call load_dotenv() on import, which would copy our .env (and its key) into os.environ.
    Fresh interpreter started in the repo, whose .env holds the key, with no key in its environment."""
    env = {name: value for name, value in os.environ.items() if name not in ("OPENAI_API_KEY", "PYTHON_DOTENV_DISABLED")}

    result = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True, cwd=REPO, env=env)

    assert result.stdout.strip() == "False"
