"""
Golden Regression Tests for Ground Temperature (TsWaveDelay) Calculations

These tests verify that the Rust TsWaveDelay implementation correctly applies
thermal inertia to ground temperature using an exponential decay model.

Formula: Lup = Tgmap0 * (1 - weight) + Tgmap1 * weight
where:  weight = exp(-33.27 * timeadd)

The decay constant 33.27 day⁻¹ corresponds to a time constant of ~43 minutes.
"""

from pathlib import Path

import numpy as np
import pytest
from solweig.rustalgos import ground

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Tolerance for ground temperature calculations
RTOL = 1e-4
ATOL = 0.1  # 0.1 units (W/m² for Lup)


@pytest.fixture
def ground_temp_inputs():
    """Load ground temperature input fixtures."""
    return {
        "gvflup": np.load(FIXTURES_DIR / "ground_temp_input_gvflup.npy"),
        "tgmap1": np.load(FIXTURES_DIR / "ground_temp_input_tgmap1.npy"),
    }


@pytest.fixture
def case1_data():
    """Load case 1: first timestep of the day."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case1.npz"))


@pytest.fixture
def case2_data():
    """Load case 2: short timestep accumulation."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case2.npz"))


@pytest.fixture
def case3_data():
    """Load case 3: long timestep."""
    return dict(np.load(FIXTURES_DIR / "ground_temp_case3.npz"))


class TestGoldenTsWaveDelay:
    """Golden tests for TsWaveDelay thermal inertia model."""

    def test_first_morning_timestep(self, ground_temp_inputs, case1_data):
        """First timestep of the day should reset previous temperature."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case1_data["input_firstdaytime"]),
            timeadd=float(case1_data["input_timeadd"]),
            timestepdec=float(case1_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case1_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 1 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case1_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 1 timeadd differs from golden fixture",
        )

        np.testing.assert_allclose(
            np.array(new_tgmap1),
            case1_data["tgmap1"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 1 Tgmap1 differs from golden fixture",
        )

    def test_short_timestep_accumulation(self, ground_temp_inputs, case2_data):
        """Short timestep should accumulate time and blend temperatures."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case2_data["input_firstdaytime"]),
            timeadd=float(case2_data["input_timeadd"]),
            timestepdec=float(case2_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case2_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 2 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case2_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 2 timeadd differs from golden fixture",
        )

    def test_long_timestep_update(self, ground_temp_inputs, case3_data):
        """Long timestep (>=59 min) should update Tgmap1."""
        lup, new_timeadd, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=int(case3_data["input_firstdaytime"]),
            timeadd=float(case3_data["input_timeadd"]),
            timestepdec=float(case3_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        np.testing.assert_allclose(
            np.array(lup),
            case3_data["lup"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 3 Lup differs from golden fixture",
        )

        np.testing.assert_allclose(
            new_timeadd,
            float(case3_data["timeadd"]),
            rtol=RTOL,
            atol=1e-6,
            err_msg="Case 3 timeadd differs from golden fixture",
        )

        np.testing.assert_allclose(
            np.array(new_tgmap1),
            case3_data["tgmap1"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="Case 3 Tgmap1 differs from golden fixture",
        )


class TestGoldenTsWaveDelayProperties:
    """Verify physical properties of TsWaveDelay model."""

    def test_thermal_inertia_effect(self, ground_temp_inputs, case2_data):
        """Output should blend between current and previous temperature."""
        lup, _, _ = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=0,
            timeadd=float(case2_data["input_timeadd"]),
            timestepdec=float(case2_data["input_timestepdec"]),
            tgmap1=ground_temp_inputs["tgmap1"],
        )
        lup_arr = np.array(lup)

        # Output should be between current and previous values
        gvflup = ground_temp_inputs["gvflup"]
        tgmap1 = ground_temp_inputs["tgmap1"]

        min_vals = np.minimum(gvflup, tgmap1)
        max_vals = np.maximum(gvflup, tgmap1)

        # Allow small numerical tolerance
        assert np.all(lup_arr >= min_vals - 0.1), "Output below minimum bound"
        assert np.all(lup_arr <= max_vals + 0.1), "Output above maximum bound"

    def test_first_morning_resets_state(self, ground_temp_inputs):
        """First morning timestep should set Tgmap1 = Tgmap0."""
        _, _, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=1,  # First morning
            timeadd=0.0,
            timestepdec=30 / 1440,
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        # After first morning, Tgmap1 should equal current input
        np.testing.assert_allclose(
            np.array(new_tgmap1),
            ground_temp_inputs["gvflup"],
            rtol=1e-5,
            err_msg="First morning should reset Tgmap1 to current value",
        )

    def test_exponential_decay_weight(self, ground_temp_inputs):
        """Verify exponential decay weight is applied correctly."""
        timeadd = 0.05  # ~72 minutes (above threshold)

        lup, _, new_tgmap1 = ground.ts_wave_delay(
            ground_temp_inputs["gvflup"],
            firstdaytime=0,
            timeadd=timeadd,
            timestepdec=timeadd,
            tgmap1=ground_temp_inputs["tgmap1"],
        )

        # Calculate expected weight
        weight = np.exp(-33.27 * timeadd)

        # Expected output
        expected = ground_temp_inputs["gvflup"] * (1 - weight) + ground_temp_inputs["tgmap1"] * weight

        np.testing.assert_allclose(
            np.array(lup),
            expected,
            rtol=1e-5,
            err_msg="Exponential decay weight not applied correctly",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
