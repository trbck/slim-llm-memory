"""Config for the Obsidian Brain: ``~/.obsidian-brain/config.toml`` + CLI overrides."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from slim_llm_memory import Embedder

DEFAULT_HOME = Path("~/.obsidian-brain")


@dataclass
class Config:
    vault: Path
    home: Path = field(default_factory=lambda: DEFAULT_HOME.expanduser())
    embedder: str = "ollama:nomic-embed-text"
    embedder_url: str = "http://localhost:11434"
    spool_drain_seconds: float = 5.0
    flush_every_changes: int = 100
    flush_every_seconds: float = 30.0
    watcher_debounce_seconds: float = 2.0
    log_level: str = "INFO"

    @property
    def index_dir(self) -> Path:
        return self.home / "index"

    @property
    def spool_dir(self) -> Path:
        return self.home / "spool"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"


_LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$')


def _parse_toml(text: str) -> dict[str, Any]:
    """tomllib when available (3.11+); otherwise a flat ``key = value`` subset."""
    try:
        import tomllib
        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if val.startswith('"') and val.endswith('"'):
            out[key] = val[1:-1]
        elif val in ("true", "false"):
            out[key] = val == "true"
        else:
            try:
                out[key] = int(val)
            except ValueError:
                out[key] = float(val)
    return out


def load_config(
    path: Path | None = None,
    *,
    vault: str | None = None,
    home: Path | None = None,
) -> Config:
    home = (home or DEFAULT_HOME).expanduser()
    path = path or (home / "config.toml")
    data: dict[str, Any] = {}
    if path.exists():
        data = _parse_toml(path.read_text(encoding="utf-8"))

    vault_str = vault or data.get("vault")
    if not vault_str:
        raise ValueError(
            f"no vault configured: pass --vault or set `vault = \"~/Vault\"` in {path}"
        )

    kwargs: dict[str, Any] = {"vault": Path(str(vault_str)).expanduser(), "home": home}
    for f in fields(Config):
        if f.name in ("vault", "home") or f.name not in data:
            continue
        kwargs[f.name] = _cast(f.name, data[f.name])
    return Config(**kwargs)


_FLOAT_FIELDS = {"spool_drain_seconds", "flush_every_seconds", "watcher_debounce_seconds"}
_INT_FIELDS = {"flush_every_changes"}


def _cast(name: str, value: Any) -> Any:
    if name in _FLOAT_FIELDS:
        return float(value)
    if name in _INT_FIELDS:
        return int(value)
    return str(value)


def make_embedder(spec: str, url: str) -> Embedder:
    """``"ollama:<model>"`` → Ollama; ``"noop:<dim>"`` → deterministic test embedder."""
    kind, _, arg = spec.partition(":")
    if kind == "ollama" and arg:
        return Embedder.ollama(arg, base_url=url)
    if kind == "noop":
        return Embedder.noop(int(arg or 384))
    raise ValueError(f"unsupported embedder spec {spec!r} (use 'ollama:<model>' or 'noop:<dim>')")
