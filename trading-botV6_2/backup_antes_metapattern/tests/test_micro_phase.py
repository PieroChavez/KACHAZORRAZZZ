import pytest

from src.core.micro_phase import MicroPhaseDetector, MicroPhase


class TestMicroPhaseDetector:
    @pytest.fixture
    def detector(self):
        return MicroPhaseDetector(lookback=5, swing_window=3, retrace_threshold=0.382, sweep_body_ratio=0.6)

    def test_indecision_on_flat_market(self, detector, df_flat):
        result = detector.detect(df_flat)
        assert result.phase == MicroPhase.INDECISION
        assert not result.allows_entry

    def test_indecision_on_insufficient_data(self, detector, df_flat):
        small = df_flat.iloc[:3]
        result = detector.detect(small)
        assert result.phase == MicroPhase.INDECISION

    def test_sweep_high_detected(self, detector, df_sweep_high):
        result = detector.detect(df_sweep_high)
        assert result.phase == MicroPhase.SWEEP_JUST_COMPLETED
        assert result.direction == "SELL"
        assert result.sweep_level is not None

    def test_sweep_low_detected(self, detector, df_sweep_low):
        result = detector.detect(df_sweep_low)
        assert result.phase == MicroPhase.SWEEP_JUST_COMPLETED
        assert result.direction == "BUY"
        assert result.sweep_level is not None

    def test_retrace_confirmed(self, detector, df_retrace_confirmed):
        result = detector.detect(df_retrace_confirmed)
        assert result.phase == MicroPhase.RETRACE_CONFIRMED
        assert result.allows_entry

    def test_bullish_fvg_detection(self, detector, df_bullish_fvg):
        """_detect_bullish_fvg returns True when c1.high < c2.low"""
        assert detector._detect_bullish_fvg(df_bullish_fvg)

    def test_bearish_fvg_detection(self, detector, df_bearish_fvg):
        """_detect_bearish_fvg returns True when c2.low > c3.high"""
        assert detector._detect_bearish_fvg(df_bearish_fvg)

    def test_no_fvg_on_flat(self, detector, df_flat):
        assert not detector._detect_bullish_fvg(df_flat)
        assert not detector._detect_bearish_fvg(df_flat)

    def test_phase_weight_mapping(self, detector):
        from src.core.micro_phase import PHASE_WEIGHT
        for phase in MicroPhase:
            assert phase in PHASE_WEIGHT

    def test_entry_allowed_phases_correct(self):
        from src.core.micro_phase import ENTRY_ALLOWED_PHASES
        allowed = {MicroPhase.SWEEP_JUST_COMPLETED, MicroPhase.FIRST_RETRACE_CANDLE,
                   MicroPhase.RETRACE_CONFIRMED, MicroPhase.IMPULSE_STARTING,
                   MicroPhase.BREAKOUT_RETEST}
        assert ENTRY_ALLOWED_PHASES == allowed
