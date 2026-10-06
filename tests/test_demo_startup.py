"""Verify empty-install seeding and protection of existing user knowledge."""
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3


def test_demo_first_start_and_preserve_existing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('start_api', root/'scripts/start_api.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shutil.copytree(root/'demo_data', tmp_path/'demo_data')
    assert module.seed_demo(tmp_path)
    index = tmp_path/'data/indexes'
    active = json.loads((index/'search/active.json').read_text())
    database = index/'search/builds'/active['build_id']/'evidence.sqlite3'
    with sqlite3.connect(database) as connection:
        assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert connection.execute('SELECT COUNT(*) FROM passages').fetchone()[0] > 0
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(database.parent/'qdrant'))
    assert client.count('support_passages').count > 0
    client.close()
    pointer = index/'search/active.json'
    pointer.write_text('{"build_id":"user-added-build"}')
    examples = index/'classification_examples.json'
    examples.write_text('user examples')
    assert not module.seed_demo(tmp_path)
    assert json.loads(pointer.read_text())['build_id'] == 'user-added-build'
    assert examples.read_text() == 'user examples'
