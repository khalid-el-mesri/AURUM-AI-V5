
from dataclasses import dataclass, asdict
import math

from risk.risk_agent import RiskAgent, RiskSnapshot


# ======================================================
# AURUM-AI — CLASSIFICATION STRATEGY ADAPTER V2
#
# Forecast Class -> Strategy Decision -> Risk Decision
#
# Academic investment simulation only.
# No real orders or financial transactions.
# ======================================================


CLASS_NAMES = {
    0: "DOWN",
    1: "NEUTRAL",
    2: "UP",
}


# ======================================================
# 1. STRATEGY DECISION
# ======================================================

@dataclass
class ClassificationStrategyDecision:

    predicted_class: str
    action: str
    reason: str

    def to_dict(self):
        return asdict(self)


# ======================================================
# 2. CLASSIFICATION STRATEGY ADAPTER
# ======================================================

class ClassificationStrategyAdapter:

    def decide(self, predicted_class, held_fraction=0.0):

        if predicted_class not in CLASS_NAMES:
            raise ValueError(
                "Invalid forecast class. "
                "Expected 0, 1 or 2."
            )

        if not math.isfinite(held_fraction):
            raise ValueError(
                "Invalid held fraction."
            )

        if not 0 <= held_fraction <= 1:
            raise ValueError(
                "Held fraction must be between 0 and 1."
            )

        label = CLASS_NAMES[predicted_class]

        # ------------------------------------------
        # UP -> BUY if no position exists
        # ------------------------------------------

        if label == "UP" and held_fraction == 0:

            action = "BUY"

            reason = (
                "UP forecast detected. "
                "No existing position. "
                "Propose opening a long position."
            )

        # ------------------------------------------
        # DOWN -> SELL if holdings exist
        # ------------------------------------------

        elif label == "DOWN" and held_fraction > 0:

            action = "SELL"

            reason = (
                "DOWN forecast detected. "
                "Existing holdings can be reduced."
            )

        # ------------------------------------------
        # NEUTRAL -> HOLD
        # ------------------------------------------

        elif label == "NEUTRAL":

            action = "HOLD"

            reason = (
                "NEUTRAL forecast detected. "
                "No new transaction proposed."
            )

        # ------------------------------------------
        # DOWN without holdings -> HOLD
        # ------------------------------------------

        elif label == "DOWN":

            action = "HOLD"

            reason = (
                "DOWN forecast detected, "
                "but no holdings are available to sell. "
                "Short selling is disabled."
            )

        # ------------------------------------------
        # UP with existing position -> HOLD
        # ------------------------------------------

        else:

            action = "HOLD"

            reason = (
                "UP forecast detected, "
                "but a position already exists. "
                "No additional position is opened."
            )

        return ClassificationStrategyDecision(
            predicted_class=label,
            action=action,
            reason=reason,
        )


# ======================================================
# 3. STRATEGY + RISK INTEGRATION TEST
# ======================================================

def main():

    print(
        "\n=== AURUM-AI | STRATEGY + RISK V2 ==="
    )

    strategy = ClassificationStrategyAdapter()

    # Reuse the existing Risk Agent V2
    risk = RiskAgent()

    # --------------------------------------------------
    # Synthetic test scenarios
    #
    # Format:
    # name, forecast_class, held_fraction,
    # current_exposure, daily_loss, drawdown
    # --------------------------------------------------

    cases = [

        (
            "UP / empty portfolio",
            2,
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        (
            "NEUTRAL / empty portfolio",
            1,
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        (
            "DOWN / empty portfolio",
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),

        (
            "DOWN / holding 5%",
            0,
            0.05,
            0.05,
            0.0,
            0.0,
        ),

        (
            "UP / daily loss 3%",
            2,
            0.0,
            0.0,
            0.03,
            0.0,
        ),

        (
            "UP / drawdown 11%",
            2,
            0.0,
            0.0,
            0.0,
            0.11,
        ),

        (
            "UP / already holding 5%",
            2,
            0.05,
            0.05,
            0.0,
            0.0,
        ),

        (
            "DOWN / holding / daily loss 3%",
            0,
            0.05,
            0.05,
            0.03,
            0.0,
        ),

    ]

    # --------------------------------------------------
    # Expected results with Risk Agent V2
    # --------------------------------------------------

    expected = [

        ("BUY", "APPROVED"),

        ("HOLD", "NO_ACTION"),

        ("HOLD", "NO_ACTION"),

        ("SELL", "APPROVED"),

        ("BUY", "REJECTED"),

        ("BUY", "REJECTED"),

        ("HOLD", "NO_ACTION"),

        # Fixed in Risk Agent V2:
        # Risk-reducing SELL must be approved.
        ("SELL", "APPROVED"),

    ]

    passed_tests = 0

    # ==================================================
    # 4. RUN INTEGRATION TESTS
    # ==================================================

    for index, case in enumerate(cases, start=1):

        (
            name,
            predicted_class,
            held,
            exposure,
            daily_loss,
            drawdown,
        ) = case

        # ----------------------------------------------
        # STEP 1: STRATEGY AGENT
        # ----------------------------------------------

        strategy_decision = strategy.decide(
            predicted_class=predicted_class,
            held_fraction=held,
        )

        # ----------------------------------------------
        # STEP 2: PROPOSE ORDER SIZE
        # ----------------------------------------------

        if strategy_decision.action == "BUY":

            order_fraction = 0.05

        elif strategy_decision.action == "SELL":

            order_fraction = min(
                held,
                0.05,
            )

        else:

            order_fraction = 0.0

        # ----------------------------------------------
        # STEP 3: CREATE RISK SNAPSHOT
        # ----------------------------------------------

        snapshot = RiskSnapshot(

            current_exposure_fraction=exposure,

            proposed_order_fraction=order_fraction,

            daily_loss_fraction=daily_loss,

            drawdown_fraction=drawdown,

            held_fraction=held,

        )

        # ----------------------------------------------
        # STEP 4: RISK AGENT V2
        # ----------------------------------------------

        risk_decision = risk.decide(

            action=strategy_decision.action,

            snapshot=snapshot,

        )

        # ----------------------------------------------
        # STEP 5: DISPLAY DECISION
        # ----------------------------------------------

        print(f"\n=== CASE {index} ===")

        print(f"Scenario: {name}")

        print(
            f"Forecast: "
            f"{strategy_decision.predicted_class}"
        )

        print(
            f"Strategy Action: "
            f"{strategy_decision.action}"
        )

        print(
            f"Strategy Reason: "
            f"{strategy_decision.reason}"
        )

        print(
            f"Proposed Order: "
            f"{order_fraction * 100:.2f}%"
        )

        print(
            f"Risk Status: "
            f"{risk_decision.status}"
        )

        print(
            f"Risk Reason: "
            f"{risk_decision.reason}"
        )

        print(
            f"Risk Timestamp: "
            f"{risk_decision.timestamp_utc}"
        )

        # ----------------------------------------------
        # STEP 6: VALIDATE EXPECTED RESULT
        # ----------------------------------------------

        actual = (

            strategy_decision.action,

            risk_decision.status,

        )

        expected_result = expected[index - 1]

        assert actual == expected_result, (
            f"Case {index} failed.\n"
            f"Expected: {expected_result}\n"
            f"Actual: {actual}"
        )

        passed_tests += 1

        print("Test Result: PASSED")

    # ==================================================
    # 5. FINAL SUMMARY
    # ==================================================

    print("\n======================================")

    print("AURUM-AI — INTEGRATION TEST SUMMARY")

    print("======================================")

    print(f"Total tests: {len(cases)}")

    print(f"Passed tests: {passed_tests}")

    print(
        f"Failed tests: "
        f"{len(cases) - passed_tests}"
    )

    print(
        "\nRisk-reducing SELL is permitted "
        "even when loss limits are reached."
    )

    print(
        "BUY orders remain subject "
        "to configured risk limits."
    )

    print("\nNo orders were executed.")

    print("No real money was used.")

    print("No test dataset was used.")

    print(
        "\nSTRATEGY + RISK V2 "
        "INTEGRATION TEST COMPLETED"
    )


if __name__ == "__main__":
    main()
