"""Internal user keys (spec 5.5): folder names never contain user-supplied strings."""
from uuid import UUID, uuid5

# Fixed forever: changing it would orphan every user's folders under DATA_ROOT.
_USER_KEY_NAMESPACE = UUID("dc1170cf-b6d1-4547-935a-62190d6a7566")


def user_key_for(user_id: str) -> str:
    """A stable, path-safe key for an upstream user_id."""
    return str(uuid5(_USER_KEY_NAMESPACE, user_id))


def checked_user_key(user_key: str) -> str:
    """`user_key` unchanged if it has exactly the form user_key_for() gives. Adapters call this before
    using a key as a folder name, so no other string can become one."""
    if not _is_canonical_uuid(user_key):
        raise ValueError(f"not a user key: {user_key!r}")
    return user_key


def _is_canonical_uuid(text: str) -> bool:
    try:
        return str(UUID(text)) == text
    except ValueError:
        return False
