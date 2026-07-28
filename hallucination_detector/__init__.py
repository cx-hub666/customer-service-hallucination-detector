"""Customer-service hallucination detection toolkit."""

from .detector import HallucinationDetector
from .evaluator import evaluate_predictions

__all__ = ["HallucinationDetector", "evaluate_predictions"]
__version__ = "1.0.0"
