# tests/test_pipeline.py

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.services.database import Base, engine, SessionLocal
from app.models import AnalysisModel

client = TestClient(app)


@pytest.fixture(autouse=True)
def prepare_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_analyze_pipeline_integration():
    payload = {"text": "I was charged $49.99 on my account. Please issue a refund."}
<<<<<<< HEAD
    
    # Send as form data instead of JSON payload
    response = client.post("/analyze", data=payload)
    assert response.status_code == 200
    
=======

    # Send as form data instead of JSON payload
    response = client.post("/analyze", data=payload)
    assert response.status_code == 200

>>>>>>> 69ed16c72529dab7f4bc2a5a87c1f156b9c04ce0
    data = response.json()
    assert data["original_text"] == payload["text"]
    assert "cleaned_text" in data
    assert "topic" in data
    assert "intent" in data
    assert "sentiment" in data

    # Confirm record is written to SQLite DB
    db = SessionLocal()
    try:
<<<<<<< HEAD
        record = db.query(AnalysisModel).filter(AnalysisModel.original_text == payload["text"]).first()
        assert record is not None
        assert record.request_id == data["request_id"]
    finally:
        db.close()
=======
        record = (
            db.query(AnalysisModel)
            .filter(AnalysisModel.original_text == payload["text"])
            .first()
        )
        assert record is not None
        assert record.request_id == data["request_id"]
    finally:
        db.close()
>>>>>>> 69ed16c72529dab7f4bc2a5a87c1f156b9c04ce0
