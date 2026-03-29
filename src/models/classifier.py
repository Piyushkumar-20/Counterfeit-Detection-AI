from dataclasses import dataclass, field
from typing import Dict
import math


@dataclass
class HybridAuthenticityClassifier:
	"""
	Lightweight logistic scorer that supports dynamic weights.
	Start here before replacing with a trained sklearn model.
	"""

	weights: Dict[str, float] = field(
		default_factory=lambda: {
			"ocr_confidence": 1.1,
			"drug_match_score": 1.5,
			"dosage_match_score": 1.0,
			"qr_validity_score": 1.2,
			"uv_similarity_score": 1.4,
			"image_match_score": 1.2,
		}
	)
	bias: float = -2.0

	def predict_proba(self, features: Dict[str, float]) -> float:
		z = self.bias
		for key, weight in self.weights.items():
			z += float(features.get(key, 0.0)) * weight
		probability = 1.0 / (1.0 + math.exp(-z))
		return min(max(probability, 0.0), 1.0)

	def weighted_score(self, features: Dict[str, float]) -> float:
		total = sum(abs(v) for v in self.weights.values()) or 1.0
		score = 0.0
		for key, weight in self.weights.items():
			score += float(features.get(key, 0.0)) * abs(weight)
		return min(max(score / total, 0.0), 1.0)
