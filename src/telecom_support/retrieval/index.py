import json
import sqlite3
from contextlib import closing
from qdrant_client import QdrantClient
from telecom_support.indexing.embeddings import LocalEmbedder
from telecom_support.indexing.vectors import COLLECTION
from telecom_support.ingestion.chunking import TOKENIZER_MODEL

class SearchIndex:
    def __init__(self, root):
        folder = root/'data/indexes/search'
        active = json.loads((folder/'active.json').read_text())['build_id']
        if len(active) != 32 or any(c not in '0123456789abcdef' for c in active):
            raise ValueError('Invalid active build ID.')
        self.path = folder/'builds'/active
        manifest = json.loads((self.path/'manifest.json').read_text())
        if manifest['model'] != TOKENIZER_MODEL or manifest['collection'] != COLLECTION:
            raise ValueError('Index embedding configuration mismatch; rebuild index.')
        self.database = self.path/'evidence.sqlite3'
        with closing(sqlite3.connect(self.database.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            self.rows = [dict(r) for r in connection.execute('SELECT * FROM passages')]
        for row in self.rows:
            row['categories'] = json.loads(row.pop('categories_json'))
            row['products'] = json.loads(row.pop('products_json'))
        self.embedder = LocalEmbedder(root/'data/indexes/embedding_model_cache')
        self.client = QdrantClient(path=str(self.path/'qdrant'))

    def close(self):
        self.client.close()
