import pytest

from src.core.md_asymmetry_filter import AsymmetryFilter


class TestAsymmetryFilter:
    @pytest.fixture
    def filter_(self):
        return AsymmetryFilter(min_candle_ratio=1.5, min_rr_ratio=3.0)

    def test_insufficient_data(self, filter_, df_flat):
        small = df_flat.iloc[:5]
        result = filter_.evaluate(small, "BUY")
        assert not result.passed
        assert result.asymmetry_factor == 0.0

    def test_buy_asymmetry_passes(self, filter_, df_asymmetric_buy):
        result = filter_.evaluate(
            df_asymmetric_buy, "BUY",
            entry_price=102.5, stop_loss=101.5, target_price=106.0,
        )
        # 5 retrace / 5 impulse = 1.0, not >= 1.5 but rr = 3.5/1.0 = 3.5 >= 3.0
        # With impulse_candles >= 2: also gets +0.2
        # candle_ratio=1.0 gets 0.2, rr=3.5 gets 0.4, impulse>=2 gets 0.2 = 0.8 >= 0.6
        assert result.passed, f"Should pass: factor={result.asymmetry_factor}, rr={result.rr_ratio}, candles={result.impulse_candles}/{result.retrace_candles}"

    def test_sell_asymmetry_passes(self, filter_, df_asymmetric_sell):
        result = filter_.evaluate(
            df_asymmetric_sell, "SELL",
            entry_price=101.5, stop_loss=102.5, target_price=98.0,
        )
        assert result.passed, f"Should pass: factor={result.asymmetry_factor}"

    def test_rr_too_low_fails(self, filter_, df_asymmetric_buy):
        result = filter_.evaluate(
            df_asymmetric_buy, "BUY",
            entry_price=102.5, stop_loss=101.0, target_price=103.0,
        )
        # rr = 0.5/1.5 = 0.33 → fails
        # candle_ratio=1.0 → 0.2, rr=0.33 → 0, impulse>=2 → 0.2 = 0.4 < 0.6
        assert not result.passed, f"Should fail: factor={result.asymmetry_factor}"

    def test_candle_ratio_contributes(self, filter_):
        import pandas as pd
        opens = [100.0] * 8 + [101.0, 101.5, 102.0, 102.5, 103.0,
                                103.5, 103.2, 102.8, 102.5, 102.0,
                                102.2, 102.5]
        closes = [100.0] * 8 + [101.5, 102.0, 102.5, 103.0, 103.5,
                                 103.0, 102.8, 102.5, 102.0, 101.8,
                                 102.5, 102.8]
        df = pd.DataFrame({"open": opens, "close": closes,
                           "high": [c + 0.3 for c in closes],
                           "low": [o - 0.3 for o in opens]})

        result = filter_.evaluate(df, "BUY")
        # Should detect 5 retrace / 5 impulse = 1.0 ratio
        assert result.impulse_candles > 0
        assert result.retrace_candles > 0

    def test_to_detection_returns_none_when_not_passed(self, filter_, df_flat):
        result = filter_.evaluate(df_flat, "BUY")
        assert result.to_detection("BUY") is None

    def test_to_detection_returns_mddetection_when_passed(self, filter_, df_asymmetric_buy):
        result = filter_.evaluate(
            df_asymmetric_buy, "BUY",
            entry_price=102.5, stop_loss=101.5, target_price=106.0,
        )
        detection = result.to_detection("BUY")
        assert detection is not None
        assert detection.concept.value == "asymmetry"
        assert detection.direction == "BUY"
        assert 0.0 < detection.confidence <= 1.0
