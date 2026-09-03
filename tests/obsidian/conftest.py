import shutil
from pathlib import Path

import pytest

from slim_llm_memory import Embedder, Memory
from slim_llm_memory.apps.obsidian.brain import Brain
from slim_llm_memory.apps.obsidian.spool import Spool

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A writable copy of the fixture vault."""
    dst = tmp_path / "vault"
    shutil.copytree(FIXTURE_VAULT, dst)
    return dst


@pytest.fixture
def brain(vault: Path, tmp_path: Path):
    home = tmp_path / "home"
    mem = Memory(home / "index", Embedder.noop(dim=64))
    b = Brain(vault, mem, Spool(home / "spool"), flush_every_changes=3, flush_every_seconds=1000)
    yield b
    b.close()
