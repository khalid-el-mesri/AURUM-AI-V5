
import math


class StrategyAgentV3:
    """Convert predicted log returns into trading decisions."""

    def __init__(self, threshold=0.002):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid signal threshold")

        self.threshold = float(threshold)

    def decide(self, predicted_return):
        if not math.isfinite(predicted_return):
            raise ValueError("Invalid model prediction")

        if predicted_return > self.threshold:
            return "LONG"

        if predicted_return < -self.threshold:
            return "SHORT"

        return "NO_TRADE"
