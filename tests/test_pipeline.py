from fastapi.testclient import TestClient

from api.server import create_app
from src.decision.decision_engine import verify


def test_decision_engine_returns_probabilistic_output():
	database = {
		"paracetamol": {"dosage": 500, "uv_required": True}
	}
	candidates = [{"name": "paracetamol", "score": 0.88}]
	qr_result = {"found": True, "decoded": True, "format_score": 0.9}
	uv_result = {"similarity": 0.85}

	result = verify(
		candidates=candidates,
		database=database,
		text="paracetamol 500 mg",
		ocr_confidence=0.9,
		qr_result=qr_result,
		uv_result=uv_result,
	)

	assert result["final_decision"] in {"authentic", "counterfeit"}
	assert 0.0 <= result["probability_authentic"] <= 1.0
	assert isinstance(result["reasoning"], list)


def test_api_health_endpoint():
	app = create_app()
	client = TestClient(app)

	response = client.get("/health")
	assert response.status_code == 200
	assert response.json()["status"] == "ok"
