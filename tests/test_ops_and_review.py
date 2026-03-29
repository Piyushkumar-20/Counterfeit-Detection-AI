from src.decision.review_queue import enqueue_for_review
from src.ops.metrics import collect_operational_metrics


def test_enqueue_for_review_creates_entry(tmp_path):
    queue_path = tmp_path / "review_queue.json"
    result = {
        "decision": {
            "drug_name": "crocin",
            "final_decision": "counterfeit",
            "confidence": 0.3,
            "probability_authentic": 0.2,
            "feature_breakdown": {"qr_validity_score": 0.0},
            "regulatory_assessment": {"category": "spurious"},
        }
    }
    row = enqueue_for_review(result, reason="unit-test", queue_path=str(queue_path))
    assert row["status"] == "pending"
    assert row["drug_name"] == "crocin"


def test_collect_operational_metrics_returns_shape():
    metrics = collect_operational_metrics()
    assert "products_count" in metrics
    assert "review_queue_count" in metrics
    assert "schema_error_count" in metrics
