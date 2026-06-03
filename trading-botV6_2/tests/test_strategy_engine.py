import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.core.strategy_engine import StrategyEngine, ScoringConfig, TradingSignal
from src.core.market_analyzer import MarketAnalyzer


def build_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "time": pd.date_range("2026-05-01", periods=len(opens), freq="15min"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class FakeProfile:
    symbol = "XAUUSDm"
    htf_context = "M15"
    htf_secondary = "M5"
    ltf_trigger = "M1"
    ltf_refine = "M1"
    min_retracement = 0.5
    fvg_entry_level = 0.5
    sl_buffer_pips = 0.5
    sl_fixed_pips = 0.0
    sl_min_pips = 10.0
    sl_max_pips = 15.0
    min_rr_ratio = 4.0
    tp_fixed_pips = 0.0
    risk_per_trade_pct = 1.0
    max_concurrent_trades = 1
    max_volume = 0
    max_daily_loss_pct = 4.0
    max_daily_trades = 15
    allowed_sessions = None
    cooldown_bars_m1 = 5


class FakeParams:
    consolidation_max_atr_ratio = 0.5
    consolidation_min_bars = 8
    expansion_tick_acceleration = 1.5
    gap_detection_pips = 5.0
    macro_event_filter = True
    min_retracement_level = 0.50
    breakout_validation = "BodyClose"
    fvg_min_size_atr_ratio = 0.2
    fvg_entry_level = 0.50
    void_min_size_pips = 3.0
    pivot_length = 8
    wyckoff_min_phase_bars = 10
    sequence_length = 3
    news_filter_active = True
    news_buffer_minutes = 30


@pytest.fixture
def weights():
    return ScoringConfig(
        htf_trend_aligned=20.0,
        in_discount_premium_zone=15.0,
        valid_market_structure=10.0,
        fvg_detected=18.0,
        order_block_valid=15.0,
        breaker_retest=12.0,
        slip_memory_present=10.0,
        equidad_sweep_confirmed=12.0,
        eslabon_breakout=15.0,
        wyckoff_phase_c_spring=20.0,
        wyckoff_phase_c_utad=20.0,
        wyckoff_phase_d_confirmed=15.0,
        ltf_bos_with_body=12.0,
        liquidity_sweep_ltf=10.0,
        in_active_session=10.0,
        no_news_event=5.0,
        regime_expansion=10.0,
        regime_not_accumulation=5.0,
        ltf_sweep_confirmation=15.0,
        multiframe_alignment=12.0,
        void_scalp_confirmed=18.0,
        triple_confluence=18.0,
        wyckoff_sos_volume=15.0,
        wyckoff_phase_d_lps=12.0,
        fvg_fresh_mitigation=8.0,
        sweep_spring_wick=10.0,
        body_close_valid=10.0,
        price_grid_aligned=8.0,
        price_establishment=10.0,
        news_high_impact_penalty=-200.0,
        wick_rejection_bonus=5.0,
        trb_manipulation_detected=12.0,
        trb_displacement=10.0,
        trb_retest=8.0,
        body_close_invalid=-50.0,
        no_sweep_detected=-30.0,
        fvg_burned_over_50=-20.0,
        spring_body_close=-40.0,
    )


@pytest.fixture
def engine(weights):
    return StrategyEngine(FakeProfile(), FakeParams(), weights, min_score=65.0, high_confidence_score=80.0)


def make_trend_data(trend="up", n=200):
    if trend == "up":
        close = 100.0 + np.arange(n) * 0.05 + np.random.rand(n) * 0.2
    elif trend == "down":
        close = 200.0 - np.arange(n) * 0.05 + np.random.rand(n) * 0.2
    else:
        close = 100.0 + np.random.rand(n) * 2
    opens = close + (np.random.rand(n) - 0.5) * 0.1
    highs = np.maximum(opens, close) + np.random.rand(n) * 0.3
    lows = np.minimum(opens, close) - np.random.rand(n) * 0.3
    return opens.tolist(), highs.tolist(), lows.tolist(), close.tolist()


def _default_timeframes(htf_df, ltf_df):
    return {"HTF": htf_df, "MID": ltf_df, "LTF": ltf_df}

class TestEvaluate:
    def test_evaluate_returns_trading_signal(self, engine):
        np.random.seed(10)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        o2, h2, l2, c2 = make_trend_data("up")
        ltf_df = build_df(o2, h2, l2, c2)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        assert isinstance(signal, TradingSignal)
        assert hasattr(signal, "direction")
        assert hasattr(signal, "score")
        assert hasattr(signal, "score_breakdown")
        assert isinstance(signal.score_breakdown, dict)

    def test_evaluate_returns_hold_with_low_score(self, engine):
        np.random.seed(11)
        o, h, l, c = make_trend_data("flat")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=True)
        if signal.direction in ("BUY", "SELL"):
            assert signal.score >= engine.min_score
        else:
            assert signal.direction == "HOLD"

    def test_evaluate_up_trend_adds_trend_bonus(self, engine):
        np.random.seed(12)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        if signal.direction == "BUY":
            breakdown = signal.score_breakdown
            trend_keys = ["htf_trend_aligned", "in_discount_zone", "valid_market_structure",
                          "regime_not_accumulation", "in_active_session"]
            for k in trend_keys:
                if k in breakdown:
                    assert breakdown[k] != 0, f"Expected non-zero for {k}"

    def test_news_penalty_applied(self, engine):
        np.random.seed(13)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        base_tf = _default_timeframes(htf_df, ltf_df)
        no_news = engine.evaluate(base_tf, now, news_active=False)
        with_news = engine.evaluate(base_tf, now, news_active=True)
        if no_news.direction == with_news.direction:
            assert with_news.score <= no_news.score
            if "news_penalty" in with_news.score_breakdown:
                assert with_news.score_breakdown["news_penalty"] <= 0

    def test_signal_has_required_fields(self, engine):
        np.random.seed(14)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        required = ["direction", "entry_price", "stop_loss", "take_profit",
                     "score", "score_breakdown", "rr_ratio", "symbol", "timestamp"]
        for field in required:
            assert hasattr(signal, field), f"Missing field: {field}"
            assert getattr(signal, field) is not None, f"Field {field} is None"

    def test_trading_signal_dataclass(self):
        now = datetime.now()
        signal = TradingSignal(
            direction="BUY", entry_price=100.0, stop_loss=99.0, take_profit=103.0,
            score=75.0, score_breakdown={"test": 10.0},
            primary_pattern=None, supporting_patterns=[],
            rr_ratio=2.0, symbol="XAUUSDm", timestamp=now,
        )
        assert signal.direction == "BUY"
        assert signal.score == 75.0
        assert signal.rr_ratio == 2.0
        assert signal.symbol == "XAUUSDm"

    def test_context_only_trade_possible(self, engine):
        np.random.seed(15)
        o, h, l, c = make_trend_data("up", n=100)
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        assert signal.notes is not None
        assert isinstance(signal.notes, list)

    def test_min_rr_filter_reduces_score(self, engine):
        np.random.seed(16)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        if signal.direction in ("BUY", "SELL") and signal.rr_ratio < engine.profile.min_rr_ratio:
            low_rr_notes = [n for n in signal.notes if "RR" in n and "bajo" in n]
            assert len(low_rr_notes) > 0

    def test_score_breakdown_is_non_empty(self, engine):
        np.random.seed(17)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        now = datetime(2026, 5, 1, 12, 0)
        timeframes = _default_timeframes(htf_df, ltf_df)
        signal = engine.evaluate(timeframes, now, news_active=False)
        if signal.direction in ("BUY", "SELL"):
            assert len(signal.score_breakdown) > 0
        total_from_breakdown = sum(v for v in signal.score_breakdown.values() if isinstance(v, (int, float)))
        assert abs(total_from_breakdown - signal.score) < 0.001 or signal.direction == "HOLD"


class TestMarketAnalyzer:
    def test_analyzer_creates_context(self):
        params = FakeParams()
        analyzer = MarketAnalyzer(params)
        np.random.seed(20)
        o, h, l, c = make_trend_data("up")
        htf_df = build_df(o, h, l, c)
        ltf_df = build_df(o, h, l, c)
        ctx = analyzer.analyze_full_context(htf_df, ltf_df)
        assert ctx is not None
        assert hasattr(ctx, "regime")
        assert hasattr(ctx, "htf_structure")
        assert hasattr(ctx, "ltf_structure")

    def test_different_trends_detectable(self):
        params = FakeParams()
        analyzer = MarketAnalyzer(params)
        for trend in ("up", "down", "flat"):
            np.random.seed(21)
            o, h, l, c = make_trend_data(trend)
            df = build_df(o, h, l, c)
            ctx = analyzer.analyze_full_context(df, df)
            assert ctx is not None


class TestScoringConfig:
    def test_defaults_match_expected(self):
        cfg = ScoringConfig()
        assert cfg.htf_trend_aligned == 20.0
        assert cfg.fvg_detected == 18.0
        assert cfg.news_high_impact_penalty == -200.0

    def test_custom_values(self):
        cfg = ScoringConfig(htf_trend_aligned=30.0, no_news_event=10.0)
        assert cfg.htf_trend_aligned == 30.0
        assert cfg.no_news_event == 10.0

    def test_all_fields_present(self):
        cfg = ScoringConfig()
        fields = ["htf_trend_aligned", "in_discount_premium_zone", "valid_market_structure",
                   "fvg_detected", "order_block_valid", "breaker_retest",
                   "news_high_impact_penalty", "body_close_invalid", "no_sweep_detected",
                   "fvg_burned_over_50", "spring_body_close", "trb_manipulation_detected",
                   "triple_confluence", "price_grid_aligned", "wick_rejection_bonus",
                   "body_close_valid", "regime_expansion", "regime_not_accumulation"]
        for f in fields:
            assert hasattr(cfg, f)
