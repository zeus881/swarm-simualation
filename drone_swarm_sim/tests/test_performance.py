"""Real-time budget checks. Run the full stress benchmark with ``python -m simulation.benchmark``."""

import pytest

from simulation.benchmark import run_case


@pytest.mark.slow
@pytest.mark.parametrize("drones", [10, 25, 50])
def test_step_fits_real_time_budget_at_30hz(drones):
    result = run_case(drones, seconds=3.0, rate=30.0)
    # p95 step time must leave headroom inside the 33.3 ms tick budget.
    assert result["step_p95_ms"] < 33.3 * 0.6, result
