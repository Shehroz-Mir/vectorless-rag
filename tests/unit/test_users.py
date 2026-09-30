from uuid import UUID

import pytest

from vectorless_rag.operations import checked_user_key, user_key_for


def test_user_key_is_stable() -> None:
    assert user_key_for("alice@example.com") == user_key_for("alice@example.com")


def test_different_users_get_different_keys() -> None:
    assert user_key_for("alice") != user_key_for("bob")


def test_user_key_is_path_safe_even_for_hostile_ids() -> None:
    key = user_key_for("../../etc/passwd")

    assert UUID(key)
    assert "/" not in key and "\\" not in key and ".." not in key


def test_a_real_user_key_passes_the_check() -> None:
    key = user_key_for("alice")

    assert checked_user_key(key) == key


@pytest.mark.parametrize("candidate", ["alice", "../x", "", "urn:uuid:" + user_key_for("a"), "{" + user_key_for("a") + "}", user_key_for("a").upper()])
def test_anything_else_is_not_a_user_key(candidate: str) -> None:
    with pytest.raises(ValueError, match="not a user key"):
        checked_user_key(candidate)
