"""PageIndex adapter: per-user clients, UserIndex and UserIndexProvider (spec 5.5)."""
import os

# PageIndex (pageindex/utils.py) and LiteLLM both call python-dotenv's load_dotenv() on import, which
# copies the nearest .env, our OpenAI key included, into os.environ for every SDK in the process.
# The key must only travel as an argument (CLAUDE.md), so switch that off before PageIndex is loaded;
# this runs first because Python imports a package before its modules. Our own settings read .env
# with dotenv_values(), which this switch does not affect.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from vectorless_rag.indexing.client_pool import PageIndexClientPool  # noqa: E402
from vectorless_rag.indexing.user_index import PageIndexUserIndex  # noqa: E402

__all__ = ["PageIndexClientPool", "PageIndexUserIndex"]
