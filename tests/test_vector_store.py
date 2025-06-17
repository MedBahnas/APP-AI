import types
from src import vector_store


def test_add_and_search(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'key')

    class DummyDocProcessor:
        def process_document(self, path):
            return {'content': 'alpha beta gamma', 'metadata': {'title': 'doc', 'type': 'text'}}
        def split_into_chunks(self, content):
            words = content.split()
            chunks = []
            start = 0
            for w in words:
                end = start + len(w)
                chunks.append({'text': w, 'start': start, 'end': end})
                start = end + 1
            return chunks

    monkeypatch.setattr(vector_store, 'DocumentProcessor', DummyDocProcessor)

    def fake_embed(self, text):
        return [float(len(text))]
    monkeypatch.setattr(vector_store.VectorStore, '_get_embedding', fake_embed)

    vs = vector_store.VectorStore(index_dir='idx')
    doc_id = vs.add_document('dummy.txt')
    assert doc_id == 'doc_1'
    assert len(vs.documents) == 3

    results = vs.search('beta', k=2)
    assert results
    assert results[0]['doc_id'] == doc_id
    assert results[0]['text'] == 'beta'

