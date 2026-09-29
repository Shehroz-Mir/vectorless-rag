from pathlib import Path

import pytest
from pydantic import ValidationError

from vectorless_rag.config import load_settings

SETTINGS_VARS = ("OPENAI_API_KEY", "CHAT_MODEL", "VISION_MODEL", "INDEX_MODEL", "MIN_IMAGE_AREA_RATIO", "RENDER_DPI")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment and .env out of these tests."""
    for name in SETTINGS_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_defaults_match_spec() -> None:
    settings = load_settings(env_file=None)

    assert settings.index_model == "gpt-5.6-luna"
    assert settings.vision_model == "gpt-5.6-luna"
    assert settings.chat_model == "gpt-5.6-sol"
    assert settings.data_root == Path("var")
    assert settings.min_image_area_ratio == 0.03
    assert settings.view_pages_image_detail == "high"
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_each_model_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_MODEL", "gpt-5.6-terra")
    monkeypatch.setenv("VISION_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("CHAT_MODEL", "gpt-5.5")

    settings = load_settings(env_file=None)

    assert (settings.index_model, settings.vision_model, settings.chat_model) == ("gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5")


def test_env_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\nCHAT_MODEL=gpt-5.5\n", encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "sk-from-file"
    assert settings.chat_model == "gpt-5.5"


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(ValidationError, match="openai_api_key"):
        load_settings(env_file=None)


def test_empty_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValidationError, match="openai_api_key"):
        load_settings(env_file=None)


def test_env_example_lists_every_setting_with_its_default() -> None:
    example = Path(__file__).resolve().parents[2] / ".env.example"
    listed = {line.split("=", 1)[0].lower() for line in example.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#")}

    from_example = load_settings(env_file=example)  # the real key still comes from the environment
    defaults = load_settings(env_file=None)

    assert listed == set(defaults.model_dump())
    assert from_example == defaults


@pytest.mark.parametrize("name, value", [("MIN_IMAGE_AREA_RATIO", "1.5"), ("RENDER_DPI", "10"), ("CHAT_MODEL", "")])
def test_invalid_values_are_rejected(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        load_settings(env_file=None)


def test_api_key_is_hidden_from_repr() -> None:
    assert "sk-test" not in repr(load_settings(env_file=None))
