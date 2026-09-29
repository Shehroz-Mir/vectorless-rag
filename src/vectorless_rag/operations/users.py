"""Internal user keys (spec 5.5): folder names never contain user-supplied strings."""
from uuid import UUID, uuid5

# Fixed forever: changing it would orphan every user's folders under DATA_ROOT.
_USER_KEY_NAMESPACE = UUID("dc1170cf-b6d1-4547-935a-62190d6a7566")


def user_key_for(user_id: str) -> str:
    """A stable, path-safe key for an upstream user_id."""
    return str(uuid5(_USER_KEY_NAMESPACE, user_id))
