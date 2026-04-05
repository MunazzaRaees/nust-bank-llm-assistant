from backend.tasks.document_ingestion import ingest_document_task
import backend.tasks.document_ingestion as ingestion_module


def test_celery_ingestion_task_upserts_chunks(monkeypatch):
    def mock_parse_document(path):
        return "Chunk one. Chunk two."

    def mock_embed_texts(chunks):
        return [[0.1, 0.2], [0.3, 0.4]]

    def mock_upsert_embeddings(**kwargs):
        return len(kwargs["payloads"])

    states = []

    def mock_update_state(**kwargs):
        states.append(kwargs)

    monkeypatch.setattr(ingestion_module, "parse_document", mock_parse_document)
    monkeypatch.setattr(ingestion_module, "embed_texts", mock_embed_texts)
    monkeypatch.setattr(ingestion_module, "upsert_embeddings", mock_upsert_embeddings)
    monkeypatch.setattr(ingest_document_task, "update_state", mock_update_state)

    result = ingest_document_task.run("policy.txt", {"source_file": "policy.txt"})

    assert len(states) > 0
    assert states[0]["state"] == "STARTED"

    assert result["status"] == "completed"
    assert result["source_file"] == "policy.txt"
    assert result["points_upserted"] == 1
