import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jangsa.store import DataStore  # noqa: E402
from jangsa.synthetic import write_all  # noqa: E402


@pytest.fixture(scope="session")
def syn_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    write_all(d)
    return d


@pytest.fixture(scope="session")
def store(syn_dir):
    s = DataStore()
    s.load_synthetic(syn_dir, include_my_store=True)
    s.load_public()
    return s
