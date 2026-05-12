import pytest
from unittest.mock import MagicMock
from src.knowledge.ingestion import DocumentIngester


@pytest.fixture
def ingester():
    store = MagicMock()
    return DocumentIngester(vector_store=store)


def test_ingest_csv(ingester, tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,value\nalpha,1\nbeta,2\n")
    count = ingester.ingest_file("client-1", str(csv_file))
    assert count >= 1
    ingester._store.add.assert_called_once()
    args = ingester._store.add.call_args
    assert args[0][0] == "client-1"


def test_ingest_txt(ingester, tmp_path):
    txt_file = tmp_path / "rules.txt"
    txt_file.write_text("NPL means non-performing loans. " * 50)
    count = ingester.ingest_file("client-1", str(txt_file))
    assert count >= 1


def test_unsupported_format_raises(ingester, tmp_path):
    bad_file = tmp_path / "file.json"
    bad_file.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        ingester.ingest_file("client-1", str(bad_file))


def test_chunking_splits_large_text(ingester):
    text = "word " * 2000
    chunks = ingester._chunk(text)
    assert len(chunks) > 1
    assert all(len(c.split()) <= ingester._chunk_size for c in chunks)
