"""S0: are the spec's default models available on this key? Prints model ids only."""
from __future__ import annotations

from openai import NotFoundError, OpenAI

from common import CHAT_MODEL, INDEX_MODEL


def main() -> None:
    client = OpenAI()
    for model in (INDEX_MODEL, CHAT_MODEL):
        try:
            found = client.models.retrieve(model)
            print(f"available: {found.id}")
        except NotFoundError:
            print(f"NOT available: {model}")
    ids = sorted(m.id for m in client.models.list() if m.id.startswith(("gpt-5", "gpt-4.1", "gpt-4o", "o3", "o4")))
    print("candidate chat/vision models on this key:")
    for model_id in ids:
        print(f"  {model_id}")


if __name__ == "__main__":
    main()
