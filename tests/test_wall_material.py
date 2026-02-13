"""Tests for wall material selection (scalar wall params from JSON)."""

from __future__ import annotations

import numpy as np
import pytest


class TestResolveWallParams:
    """Test the resolve_wall_params() function directly."""

    def test_brick_returns_correct_params(self):
        from solweig.loaders import resolve_wall_params

        tgk, tstart, tmaxlst = resolve_wall_params("brick")
        assert tgk == pytest.approx(0.40)
        assert tstart == pytest.approx(-4.0)
        assert tmaxlst == pytest.approx(15.0)

    def test_concrete_returns_correct_params(self):
        from solweig.loaders import resolve_wall_params

        tgk, tstart, tmaxlst = resolve_wall_params("concrete")
        assert tgk == pytest.approx(0.35)
        assert tstart == pytest.approx(-5.0)
        assert tmaxlst == pytest.approx(16.0)

    def test_wood_returns_correct_params(self):
        from solweig.loaders import resolve_wall_params

        tgk, tstart, tmaxlst = resolve_wall_params("wood")
        assert tgk == pytest.approx(0.50)
        assert tstart == pytest.approx(-2.0)
        assert tmaxlst == pytest.approx(14.0)

    def test_cobblestone_returns_default_params(self):
        from solweig.loaders import resolve_wall_params

        tgk, tstart, tmaxlst = resolve_wall_params("cobblestone")
        assert tgk == pytest.approx(0.37)
        assert tstart == pytest.approx(-3.41)
        assert tmaxlst == pytest.approx(15.0)

    def test_case_insensitive(self):
        from solweig.loaders import resolve_wall_params

        for name in ("Brick", "BRICK", "bRiCk"):
            tgk, _, _ = resolve_wall_params(name)
            assert tgk == pytest.approx(0.40), f"Failed for {name!r}"

    def test_invalid_material_raises_valueerror(self):
        from solweig.loaders import resolve_wall_params

        with pytest.raises(ValueError, match="Unknown wall material"):
            resolve_wall_params("marble")

    def test_error_message_lists_valid_options(self):
        from solweig.loaders import resolve_wall_params

        with pytest.raises(ValueError, match="brick") as exc_info:
            resolve_wall_params("unknown")
        msg = str(exc_info.value)
        assert "concrete" in msg
        assert "wood" in msg
        assert "cobblestone" in msg

    def test_with_custom_materials(self):
        from types import SimpleNamespace

        from solweig.loaders import resolve_wall_params
        from solweig.utils import dict_to_namespace

        raw = dict_to_namespace(
            {
                "Ts_deg": {"Value": {"Brick_wall": 0.99}},
                "Tstart": {"Value": {"Brick_wall": -1.0}},
                "TmaxLST": {"Value": {"Brick_wall": 13.0}},
            }
        )
        assert isinstance(raw, SimpleNamespace)
        tgk, tstart, tmaxlst = resolve_wall_params("brick", materials=raw)
        assert tgk == pytest.approx(0.99)
        assert tstart == pytest.approx(-1.0)
        assert tmaxlst == pytest.approx(13.0)


class TestWallMaterialInCalculate:
    """Test wall_material parameter in the full calculate() pipeline."""

    @pytest.fixture
    def simple_inputs(self):
        """Minimal inputs for a daytime calculation."""
        from datetime import datetime

        from solweig import HumanParams, Location, SurfaceData, Weather

        dsm = np.full((3, 3), 2.0, dtype=np.float32)
        surface = SurfaceData(dsm=dsm)
        surface.compute_svf()
        location = Location(latitude=57.7, longitude=12.0, utc_offset=1)
        weather = Weather(
            datetime=datetime(2023, 7, 15, 12, 0),
            ta=25.0,
            rh=50.0,
            global_rad=800.0,
        )
        human = HumanParams()
        return surface, location, weather, human

    def test_wall_material_none_uses_default(self, simple_inputs):
        """wall_material=None should produce same result as no param."""
        from solweig import calculate

        surface, location, weather, human = simple_inputs
        result_default = calculate(surface, location, weather, human=human, use_anisotropic_sky=False)
        result_none = calculate(surface, location, weather, human=human, wall_material=None, use_anisotropic_sky=False)

        np.testing.assert_array_equal(result_default.tmrt, result_none.tmrt)

    def test_brick_differs_from_default(self, simple_inputs):
        """Brick wall material should produce different Tmrt than default."""
        from solweig import calculate

        surface, location, weather, human = simple_inputs
        # Use isotropic sky — this flat surface has no explicit wall pixels,
        # but wall material parameters still affect ground temperature through
        # the isotropic radiation pathway (tgk_wall / tstart_wall scalars).
        result_default = calculate(surface, location, weather, human=human, use_anisotropic_sky=False)
        result_brick = calculate(
            surface, location, weather, human=human, wall_material="brick", use_anisotropic_sky=False
        )

        assert not np.array_equal(result_default.tmrt, result_brick.tmrt), (
            "Brick wall material should produce different Tmrt than default"
        )

    def test_wood_higher_wall_temp_than_brick(self, simple_inputs):
        """Wood (TgK=0.50) should produce higher wall temp than brick (TgK=0.40)."""
        from solweig.components.ground import compute_ground_temperature

        surface, location, weather, _ = simple_inputs
        weather.compute_derived(location)

        alb = np.full((3, 3), 0.15, dtype=np.float32)
        emis = np.full((3, 3), 0.95, dtype=np.float32)
        tgk = np.full((3, 3), 0.37, dtype=np.float32)
        tstart = np.full((3, 3), -3.41, dtype=np.float32)
        tmaxlst = np.full((3, 3), 15.0, dtype=np.float32)

        gb_wood = compute_ground_temperature(
            weather,
            location,
            alb,
            emis,
            tgk,
            tstart,
            tmaxlst,
            tgk_wall=0.50,
            tstart_wall=-2.0,
            tmaxlst_wall=14.0,
        )
        gb_brick = compute_ground_temperature(
            weather,
            location,
            alb,
            emis,
            tgk,
            tstart,
            tmaxlst,
            tgk_wall=0.40,
            tstart_wall=-4.0,
            tmaxlst_wall=15.0,
        )

        assert gb_wood.tg_wall > gb_brick.tg_wall, (
            f"Wood tg_wall ({gb_wood.tg_wall:.2f}) should exceed brick ({gb_brick.tg_wall:.2f})"
        )
