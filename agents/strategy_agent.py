
from dataclasses import dataclass, asdict
import math


# AURUM-AI — STRATEGY AGENT
# Academic simulation only.
# No real orders or financial transactions.


@dataclass
class StrategyDecision:
    action: str
    predicted_return: float
    threshold: float
    reason: str

    def to_dict(self):
        return asdict(self)


class StrategyAgent:

    def __init__(self, threshold=0.0003):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Threshold must be positive.")

        self.threshold = threshold

    def decide(self, predicted_return):

        if not math.isfinite(predicted_return):
            raise ValueError("Invalid predicted return.")

        if predicted_return > self.threshold:
            action = "BUY"
            reason = "Positive forecast above threshold."

        elif predicted_return < -self.threshold:
            action = "SELL"
            reason = "Negative forecast below threshold."

        else:
            action = "HOLD"
            reason = "Forecast magnitude is too small."

        return StrategyDecision(
            action=action,
            predicted_return=float(predicted_return),
            threshold=self.threshold,
            reason=reason,
        )


if __name__ == "__main__":

    agent = StrategyAgent(threshold=0.0003)

    # Illustrative examples, NOT real predictions.
    examples = [
        0.00002,
        -0.00011,
        0.00050,
        -0.00060,
    ]

    print("\n=== AURUM-AI | STRATEGY AGENT ===")

    for prediction in examples:
        result = agent.decide(prediction)

        print(
            f"\nForecast: {prediction:+.6f}"
            f"\nDecision: {result.action}"
            f"\nReason: {result.reason}"
        )

    print("\nSTRATEGY AGENT TEST COMPLETED")
