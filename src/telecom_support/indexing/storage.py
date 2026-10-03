"""SQLite is the complete record store; passage IDs link it to Qdrant."""
import json
import sqlite3
from contextlib import closing

def write_database(path, records, passages, manifest):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY, source_type TEXT NOT NULL,
                source_file TEXT NOT NULL, source_version TEXT NOT NULL,
                record_json TEXT NOT NULL);
            CREATE TABLE passages (
                passage_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                source_type TEXT NOT NULL, text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                categories_json TEXT NOT NULL, products_json TEXT NOT NULL);
            CREATE INDEX passages_source ON passages(source_id);
            CREATE TABLE build_metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        """)
        connection.executemany("INSERT INTO sources VALUES (?, ?, ?, ?, ?)", [
            (r.source_id, r.source_type, r.source_file, r.source_version, r.model_dump_json())
            for r in records])
        connection.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (p['passage_id'], p['source_id'], p['source_type'], p['text'], p['token_count'],
             json.dumps(p['categories']), json.dumps(p['products'])) for p in passages])
        connection.execute("INSERT INTO build_metadata VALUES (?, ?)",
                           ("manifest", json.dumps(manifest)))
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Broken SQL source references.")

def fetch_evidence(path, passage_id):
    """Later retrieval uses a Qdrant point ID to fetch text and its full source."""
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        row = connection.execute("""SELECT p.text, s.record_json FROM passages p
            JOIN sources s ON s.source_id=p.source_id WHERE p.passage_id=?""",
            (passage_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown passage ID: {passage_id}")
    return {"passage_text": row[0], "source": json.loads(row[1])}
