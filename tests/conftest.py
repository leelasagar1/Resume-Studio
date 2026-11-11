"""Tests never touch the real data directory (cached job analyses, run log)."""
import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'data'))
    from app import review
    review._MEMORY.clear()
