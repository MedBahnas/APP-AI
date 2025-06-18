from src.document_processor import DocumentProcessor


def test_split_into_chunks_basic():
    text = "ABCDEFGHIJK"
    chunks = DocumentProcessor.split_into_chunks(None, text, chunk_size=5, overlap=2)
    assert [c["text"] for c in chunks] == ["ABCDE", "DEFGH", "GHIJK"]
    assert [c["start"] for c in chunks] == [0, 3, 6]

