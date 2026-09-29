from uuid import UUID

from vectorless_rag.operations import user_key_for


def test_user_key_is_stable() -> None:
    assert user_key_for("alice@example.com") == user_key_for("alice@example.com")


def test_different_users_get_different_keys() -> None:
    assert user_key_for("alice") != user_key_for("bob")


def test_user_key_is_path_safe_even_for_hostile_ids() -> None:
    key = user_key_for("../../etc/passwd")

    assert UUID(key)
    assert "/" not in key and "\\" not in key and ".." not in key
