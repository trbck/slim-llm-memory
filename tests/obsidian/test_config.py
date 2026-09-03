from pathlib import Path

import pytest

from slim_llm_memory.apps.obsidian.config import Config, load_config, make_embedder


def test_defaults_when_no_file(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml", vault=str(tmp_path / "Vault"), home=tmp_path / "home")
    assert cfg.vault == (tmp_path / "Vault")
    assert cfg.embedder == "ollama:nomic-embed-text"
    assert cfg.spool_drain_seconds == 5.0
    assert cfg.index_dir == tmp_path / "home" / "index"
    assert cfg.spool_dir == tmp_path / "home" / "spool"
    assert cfg.logs_dir == tmp_path / "home" / "logs"


def test_file_values_and_cli_override(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        'vault = "~/Vault"\nembedder = "noop:64"\nspool_drain_seconds = 1\n'
        'flush_every_changes = 7\nlog_level = "DEBUG"\n',
        encoding="utf-8",
    )
    cfg = load_config(p, home=tmp_path)
    assert cfg.vault == Path("~/Vault").expanduser()
    assert cfg.embedder == "noop:64"
    assert cfg.spool_drain_seconds == 1.0
    assert cfg.flush_every_changes == 7
    assert cfg.log_level == "DEBUG"
    # explicit vault wins over file
    cfg2 = load_config(p, vault=str(tmp_path / "Other"), home=tmp_path)
    assert cfg2.vault == tmp_path / "Other"


def test_missing_vault_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="vault"):
        load_config(tmp_path / "missing.toml", home=tmp_path)


def test_make_embedder():
    e = make_embedder("noop:32", "http://unused")
    assert e.name == "noop:32" and e.dim == 32
    o = make_embedder("ollama:nomic-embed-text", "http://localhost:11434")
    assert o.name == "ollama:nomic-embed-text"
    with pytest.raises(ValueError):
        make_embedder("gemini:x", "")
