"""SymbolEvaluator — Estrategia de evaluación por símbolo

Extraído de TradingBot._evaluate_symbol para reducir complejidad de bot.py
y permitir testeo aislado del pipeline de decisión por símbolo.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

import pandas as pd
from loguru import logger

from src.core.market_map import MarketMapResult
from src.executor.order_types import OrderType
from src.utils.helpers import is_in_session

if TYPE_CHECKING:
    from src.bot import TradingBot


class SymbolEvaluator:
    """Evalúa un símbolo dentro del pipeline del bot.

    Delega en los subsistemas del bot (inyectado como ``bot``)
    para ejecutar el pipeline completo: señal → convicción → risk → ejecución.

    El método principal ``evaluate()`` replica exactamente la lógica
    de ``TradingBot._evaluate_symbol()``.
    """

    def _record_skipped(self, bot: TradingBot, symbol: str, signal,
                         conv: float, regime, session_profile, ltf_df,
                         reason: str, atr_val: float, pip_value: float):
        """Helper para registrar señal saltada con tracking de outcome."""
        if not hasattr(bot, 'meta_learner') or not bot.meta_learner:
            return
        regime_name = regime.regime.value if regime else "UNKNOWN"
        session_label = session_profile.label if session_profile else "UNKNOWN"
        pattern_name = signal.primary_pattern.type.name if signal.primary_pattern else None
        price = ltf_df["close"].iloc[-1] if ltf_df is not None else None
        bot.meta_learner.record_skipped_signal(
            symbol=symbol, direction=signal.direction,
            score=signal.score, conviction=conv,
            regime=regime_name, session=session_label,
            reason=reason, pattern_type=pattern_name, price=price,
            entry_price=signal.entry_price, atr_val=atr_val, pip_value=pip_value,
        )

    def evaluate(self, bot: TradingBot, symbol: str, sym_data: dict, dxy_df: pd.DataFrame = None):
        """Pipeline completo de evaluación por símbolo."""
        # ── acceso a estado interno del bot ──────────────────────────────────
        bot._cycle_cache.clear()
        bot._positions_cache = []

        profile = sym_data["profile"]
        engine = sym_data["engine"]
        pip_value = sym_data["pip_value"]

        timeframes = bot.fetcher.get_dataframes(symbol, count=300)
        if len(timeframes) < 3:
            return

        ltf_df = None
        for tf in ["1min", "3min", "5min"]:
            df = timeframes.get(tf)
            if df is not None and len(df) >= 50:
                ltf_df = df
                break
        if ltf_df is None:
            return

        last_close = ltf_df["close"].iloc[-1]
        last_time = ltf_df.index[-1] if hasattr(ltf_df.index, "values") else datetime.now(timezone.utc)
        bot.correlation_engine.update(symbol, last_time, last_close)

        bot._manage_pending_orders(symbol, ltf_df)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        news_active = bot.news_calendar.is_high_impact_active(now, symbol)

        selection = bot.tf_optimizer.analyze(symbol, timeframes)
        engine.tf_groups = selection.groups
        htf_df = timeframes.get(selection.htf)
        ltf_df = timeframes.get(selection.ltf)
        if htf_df is None or len(htf_df) <= 20:
            htf_df = engine._pick_tf(timeframes, "HTF")
        if ltf_df is None or len(ltf_df) <= 20:
            ltf_df = engine._pick_tf(timeframes, "LTF")

        # ── Poll skipped outcomes (no modifica operativa) ──
        if hasattr(bot, 'meta_learner') and bot.meta_learner:
            try:
                bot.meta_learner.poll_skipped_outcomes(symbol, ltf_df["close"].iloc[-1])
            except Exception:
                pass

        regime = bot.regime_detectors[symbol].detect(htf_df, ltf_df)
        bot._last_regime[symbol] = regime

        signal = engine.evaluate_adaptive(
            timeframes, now, news_active=news_active,
            regime=regime, htf_df=htf_df, ltf_df=ltf_df, dxy_df=dxy_df,
        )

        if selection.volatility_regime != "MEDIUM":
            logger.info(f"[{symbol}] TF optimizer: vol={selection.volatility_regime}, "
                        f"HTF={selection.htf}, LTF={selection.ltf}" +
                        (f", degradados={selection.degraded_tfs}" if selection.degraded_tfs else ""))

        # ── Enhanced Order Flow & DOM analysis (Mejora 13 — Modo Experto) ──
        from src.utils.helpers import atr as _atr_fn
        of_atr = _atr_fn(ltf_df, 14).iloc[-1] if ltf_df is not None and len(ltf_df) >= 14 else 0.0
        of_signal = bot.order_flow.analyze(symbol, ltf_df, atr_val=of_atr, pip=pip_value)
        of_bonus, of_reason = bot.order_flow.get_signal_contribution(of_signal, signal.direction)

        if of_bonus != 0:
            signal.score = min(100.0, signal.score + of_bonus)
            signal.notes.append(f"OrderFlow {signal.direction}: {of_reason} ({of_bonus:+.1f})")
            logger.info(f"[{symbol}] OrderFlow bonus: {of_bonus:+.1f} ({of_reason})")

        if bot.expert_mode and of_signal.notes:
            for note in of_signal.notes[:4]:
                logger.info(f"[{symbol}] OrderFlow: {note}")
            if of_signal.absorption_active and of_signal.absorption_clusters:
                cluster_prices = [c.poc for c in of_signal.absorption_clusters[:3]]
                logger.info(f"[{symbol}] Absorption clusters @ {cluster_prices}")
            if of_signal.iceberg_detected:
                logger.info(f"[{symbol}] Iceberg: {len(of_signal.iceberg_levels)} levels, "
                            f"{of_signal.iceberg_chunks} chunk repetitions")
            if of_signal.delta_macd:
                dm = of_signal.delta_macd
                if dm.divergence_bullish or dm.divergence_bearish:
                    div_type = "BULL" if dm.divergence_bullish else "BEAR"
                    logger.info(f"[{symbol}] Delta-MACD {div_type} divergence (hist={dm.histogram:.4f})")
            if of_signal.exhaustion_active:
                logger.info(f"[{symbol}] Exhaustion: {of_signal.exhaustion_side}")

        bot._signal_id_counter += 1
        signal_id = f"{symbol}_{int(time.time())}_{bot._signal_id_counter}"

        conv = signal.conviction

        if of_bonus != 0 and signal.direction in ("BUY", "SELL"):
            old_conv = conv
            conv = min(1.0, max(0.01, conv * (1.0 + of_bonus / 100.0)))
            if conv != old_conv:
                logger.info(f"[{symbol}] OrderFlow ajusta convicción: {old_conv:.0%} → {conv:.0%}")

        session_label = signal.session_profile.label if signal and signal.session_profile else ""
        dxy_trend_val = bot._last_dxy_trend if hasattr(bot, '_last_dxy_trend') else ""
        atr_r = regime.atr_ratio if regime else 1.0
        vol_regime_val = "HIGH" if atr_r > 1.5 else "LOW" if atr_r < 0.7 else "MEDIUM"
        streak_val = -bot.adaptive_cooldown.get_streak(symbol).consecutive_losses

        ctx_vector = bot.context_oracle.build_context_vector(
            adx=regime.adx_value if regime else 0,
            atr_ratio=regime.atr_ratio if regime else 1.0,
            regime_type=regime.regime.value if regime else "TRANSITION",
            alignment=regime.trend_alignment if regime else "NEUTRAL",
            hour=now.hour,
            conviction=conv,
            score_net=signal.distribution.mean if signal.distribution else signal.score,
            session=session_label,
            day_of_week=now.weekday(),
            volatility_regime=vol_regime_val,
            dxy_trend=dxy_trend_val,
            streak=streak_val,
        )

        if signal.direction == "BUY":
            oracle_buy_score = signal.score + 1
            oracle_sell_score = signal.score
        elif signal.direction == "SELL":
            oracle_buy_score = signal.score
            oracle_sell_score = signal.score + 1
        else:
            oracle_buy_score = signal.score
            oracle_sell_score = signal.score
        oracle_dir, oracle_confidence, oracle_wr = bot.context_oracle.predict_direction(
            ctx_vector,
            oracle_buy_score,
            oracle_sell_score,
            signal.score_breakdown if signal.direction == "BUY" else {},
            signal.score_breakdown if signal.direction == "SELL" else {},
            regime=regime.regime.value if regime else "",
            conviction=conv,
        )
        if oracle_dir != "HOLD" and oracle_confidence > 0.5:
            logger.info(f"[{symbol}] Oracle sugiere {oracle_dir} "
                        f"(conf={oracle_confidence:.0%}, WR={oracle_wr:.0%}) "
                        f"vs Signal: {signal.direction}")

        logger.info(f"[{symbol}] {signal.direction} | Score: {signal.score:.1f} | "
                    f"Convicción: {conv:.0%} | "
                    f"Primary: {signal.primary_pattern.type.name if signal.primary_pattern else 'none'}"
                    f"{' | Notes: '+'; '.join(signal.notes) if signal.notes else ''}")
        if signal.distribution:
            logger.info(f"[{symbol}] Distribución: μ={signal.distribution.mean:.1f} σ={signal.distribution.std:.1f} "
                        f"convergencia={signal.distribution.convergence:.0%}")
        if signal.regime_context:
            logger.info(f"[{symbol}] Régimen: {signal.regime_context.regime.value} "
                        f"(conf={signal.regime_context.confidence:.0%}, ADX={signal.regime_context.adx_value:.0f})")

        # --- Alimentar market_memory con swing points detectados ---
        from src.utils.helpers import atr, find_swing_points
        atr_val = atr(ltf_df, 14).iloc[-1]
        highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)
        current_close = ltf_df["close"].iloc[-1]
        for idx in highs_idx[-10:]:
            price = round(ltf_df["high"].iloc[idx], 5)
            outcome = "break" if current_close > price else "bounce"
            bot.market_memory.record_interaction(symbol, price, outcome, ltf_df=ltf_df)
        for idx in lows_idx[-10:]:
            price = round(ltf_df["low"].iloc[idx], 5)
            outcome = "break" if current_close < price else "bounce"
            bot.market_memory.record_interaction(symbol, price, outcome, ltf_df=ltf_df)

        # --- Ajustar convicción según niveles clave del market_memory ---
        level_bias = bot.market_memory.get_level_bias(symbol, current_close, atr_val)
        if level_bias and signal.direction in ("BUY", "SELL"):
            old_conv = conv
            if level_bias == "BULLISH_BIAS" and signal.direction == "SELL":
                conv *= 0.5
                logger.info(f"[{symbol}] Soporte cerca (bias BULLISH): SELL penalizado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BULLISH_BIAS" and signal.direction == "BUY":
                conv = min(1.0, conv * 1.2)
                logger.info(f"[{symbol}] Soporte cerca (bias BULLISH): BUY potenciado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BEARISH_BIAS" and signal.direction == "BUY":
                conv *= 0.5
                logger.info(f"[{symbol}] Resistencia cerca (bias BEARISH): BUY penalizado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BEARISH_BIAS" and signal.direction == "SELL":
                conv = min(1.0, conv * 1.2)
                logger.info(f"[{symbol}] Resistencia cerca (bias BEARISH): SELL potenciado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "RANGE_BIAS":
                conv *= 0.8
                logger.info(f"[{symbol}] Rango detectado: convicción reducida {old_conv:.0%} → {conv:.0%}")

        htf_alignment = regime.trend_alignment if regime else "NEUTRAL"
        adx_value = regime.adx_value if regime else 0
        regime_confidence = regime.confidence if regime else 0
        if htf_alignment in ("BULLISH_ALIGNED", "BEARISH_ALIGNED") and signal.direction in ("BUY", "SELL"):
            is_aligned = (htf_alignment == "BULLISH_ALIGNED" and signal.direction == "BUY") or \
                         (htf_alignment == "BEARISH_ALIGNED" and signal.direction == "SELL")
            adx_factor = min(adx_value / 50.0, 1.0)
            conf_factor = 0.3 + (regime_confidence * 0.7)
            strength = min(1.0, adx_factor * conf_factor)
            old_conv = conv
            if is_aligned:
                mult = 1.0 + strength * 1.5
                conv = min(1.0, conv * mult)
                logger.info(f"[{symbol}] Tendencia {htf_alignment} (ADX={adx_value:.0f}, "
                            f"conf={regime_confidence:.0%}): {signal.direction} "
                            f"potenciado {old_conv:.0%} → {conv:.0%} (×{mult:.1f})")
            else:
                mult = max(0.50, 1.0 - strength * 0.50)
                conv *= mult
                logger.info(f"[{symbol}] Tendencia {htf_alignment} (ADX={adx_value:.0f}, "
                            f"conf={regime_confidence:.0%}): {signal.direction} "
                            f"contra-tendencia {old_conv:.0%} → {conv:.0%} (×{mult:.2f})")
        elif htf_alignment in ("HTF_BULLISH_LTF_BEARISH", "HTF_BEARISH_LTF_BULLISH") and signal.direction in ("BUY", "SELL"):
            htf_dir = "BUY" if htf_alignment == "HTF_BULLISH_LTF_BEARISH" else "SELL"
            is_aligned = signal.direction == htf_dir
            adx_factor = min(adx_value / 40.0, 1.0)
            conf_factor = 0.5 + (regime_confidence * 0.5)
            strength = min(1.0, adx_factor * conf_factor)
            old_conv = conv
            if is_aligned:
                mult = 1.0 + strength * 0.8
                conv = min(1.0, conv * mult)
                logger.info(f"[{symbol}] HTF {htf_dir} (ADX={adx_value:.0f}): {signal.direction} "
                            f"potenciado {old_conv:.0%} → {conv:.0%} (×{mult:.1f})")
            else:
                mult = max(0.60, 1.0 - strength * 0.50)
                conv *= mult
                logger.info(f"[{symbol}] HTF en contra ({htf_dir}, ADX={adx_value:.0f}): "
                            f"{signal.direction} penalizado {old_conv:.0%} → {conv:.0%} (×{mult:.2f})")

        # ── Timeframe Consensus adjustment ──
        if signal.consensus and signal.direction in ("BUY", "SELL"):
            alignment = signal.consensus.alignment_with(signal.direction)
            if alignment < 0.3:
                old_conv = conv
                conv *= 0.5
                logger.info(f"[{symbol}] Consensus TF {signal.consensus.overall_direction} "
                            f"vs {signal.direction}: convicción {old_conv:.0%} → {conv:.0%}")
            elif alignment > 0.7:
                old_conv = conv
                conv = min(1.0, conv * 1.3)
                logger.info(f"[{symbol}] Consensus TF {signal.consensus.overall_direction} "
                            f"a favor: convicción {old_conv:.0%} → {conv:.0%}")
            if signal.consensus.tf_votes:
                active_dirs = {}
                for v in signal.consensus.tf_votes:
                    if v.direction != "HOLD":
                        active_dirs[v.tf_name] = v.direction
                if active_dirs:
                    logger.info(f"[{symbol}] Votos TF: {active_dirs}")

        session_profile = bot.session_profiler.profile(symbol, ltf_df, now)

        # ── Market Map Scalping Pre-filter ──
        market_map = bot.market_maps.get(symbol)
        market_result: Optional[MarketMapResult] = None
        if market_map is not None and signal.direction in ("BUY", "SELL"):
            try:
                market_result = market_map.evaluate(timeframes, direction=signal.direction, conviction=conv)
                bot._last_md_detections[symbol] = market_result.md_detections
                if market_result.decision == "NO_TRADE":
                    logger.info(f"[{symbol}] MarketMap: NO_TRADE ({market_result.notes[-1] if market_result.notes else ''})")
                    self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "market_map_no_trade", atr_val, pip_value)
                    return
                if market_result.decision == "WAIT":
                    logger.info(f"[{symbol}] MarketMap: WAIT ({market_result.notes[-1] if market_result.notes else ''})")
                    if market_result.phase and not market_result.phase.allows_entry:
                        return
                if market_result.decision == "TRADE":
                    bonus = market_result.score_bonus + (market_result.confidence * 20)
                    signal.score = min(100.0, signal.score + bonus)
                    signal.notes.append(f"MarketMap {market_result.direction}: +{bonus:.1f} ({len(market_result.notes)} checks)")
                    logger.info(f"[{symbol}] MarketMap: TRADE {market_result.direction} (bonus={bonus:.1f}, confidence={market_result.confidence:.0%})")
            except Exception as e:
                logger.warning(f"[{symbol}] MarketMap error: {e}")

        # ── MicroPredictor: proyección combinada OrderFlow + confluencias + MarketMap ──
        if market_result is not None:
            try:
                pred = bot.micro_predictor.predict(
                    of_signal=of_signal,
                    market_map=market_result.market_map,
                    market_map_result=market_result,
                    route=market_result.route,
                    regime=regime,
                    current_price=ltf_df["close"].iloc[-1] if ltf_df is not None else 0.0,
                    atr_val=atr_val,
                    pip=pip_value,
                )
                bot._last_prediction[symbol] = pred
                if not pred.is_neutral:
                    logger.info(f"[{symbol}] Predicción: {pred.summary}")
                    if pred.confidence >= 0.6:
                        signal.notes.append(f"Predictor {pred.direction}: {pred.primary_reason} "
                                            f"(conf={pred.confidence:.0%}, ~{pred.estimated_bars_m1}barras)")
                        if pred.direction == signal.direction and pred.confidence >= 0.65:
                            boost = pred.confidence * 8
                            signal.score = min(100.0, signal.score + boost)
                            conv = min(1.0, conv * (1.0 + pred.confidence * 0.2))
                            logger.info(f"[{symbol}] Predictor refuerza señal: score+{boost:.1f}, conv+{pred.confidence*20:.0%}")
                else:
                    logger.info(f"[{symbol}] Predicción: NEUTRAL")
            except Exception as e:
                logger.debug(f"[{symbol}] MicroPredictor error: {e}")

        # ── Bayesian Ensemble: fusión de expertos (no bloquea) ──
        mm_conf = market_result.confidence if market_result is not None else 0.0
        of_conf = min(1.0, abs(of_bonus) / 30.0) if of_bonus != 0 else 0.0
        last_pred = bot._last_prediction.get(symbol) if market_result is not None else None
        micro_conf = last_pred.confidence if (last_pred and not last_pred.is_neutral) else 0.0
        align_val = 0.5
        if regime:
            if htf_alignment in ("BULLISH_ALIGNED", "BEARISH_ALIGNED"):
                is_aligned = (htf_alignment == "BULLISH_ALIGNED" and signal.direction == "BUY") or \
                             (htf_alignment == "BEARISH_ALIGNED" and signal.direction == "SELL")
                adx_factor = min(adx_value / 50.0, 1.0)
                conf_factor = 0.3 + (regime_confidence * 0.7)
                strength = min(1.0, adx_factor * conf_factor)
                align_val = 0.5 + strength * 0.5 if is_aligned else 0.5 - strength * 0.4
            elif htf_alignment in ("HTF_BULLISH_LTF_BEARISH", "HTF_BEARISH_LTF_BULLISH"):
                htf_dir = "BUY" if htf_alignment == "HTF_BULLISH_LTF_BEARISH" else "SELL"
                is_aligned = signal.direction == htf_dir
                adx_factor = min(adx_value / 40.0, 1.0)
                conf_factor = 0.5 + (regime_confidence * 0.5)
                strength = min(1.0, adx_factor * conf_factor)
                align_val = 0.5 + strength * 0.3 if is_aligned else 0.5 - strength * 0.3
        bot._last_market_map_conf = getattr(bot, '_last_market_map_conf', {})
        bot._last_market_map_conf[symbol] = mm_conf
        bot._last_of_conf = getattr(bot, '_last_of_conf', {})
        bot._last_of_conf[symbol] = of_conf
        bot._last_micro_conf = getattr(bot, '_last_micro_conf', {})
        bot._last_micro_conf[symbol] = micro_conf
        bot._last_regime_alignment = getattr(bot, '_last_regime_alignment', {})
        bot._last_regime_alignment[symbol] = align_val

        if hasattr(bot, 'bayesian_ensemble'):
            try:
                ensemble_result = bot.bayesian_ensemble.evaluate(
                    symbol=symbol,
                    direction=signal.direction,
                    regime=regime.regime.value if regime else "UNKNOWN",
                    market_map_conf=mm_conf,
                    of_confidence=of_conf,
                    micro_conf=micro_conf,
                    regime_alignment=align_val,
                    raw_conviction=conv,
                )
                old_conv = conv
                conv = min(1.0, max(0.01, ensemble_result["adjusted_conviction"]))
                logger.info(
                    f"[{symbol}] Bayesian: conv {old_conv:.0%} → {conv:.0%} "
                    f"(uncert={ensemble_result['uncertainty']:.2f}, "
                    f"ens_conf={ensemble_result.get('ensemble_conf', 0):.2f}, "
                    f"pesos={ensemble_result['expert_weights']})"
                )
            except Exception as e_be:
                logger.debug(f"[{symbol}] BayesianEnsemble error (non-blocking): {e_be}")

        # ── Session Auto-Tuning ──
        bot._session_vol_mult = session_profile.volume_adjustment
        bot._session_scale_n = getattr(profile, 'scale_entries', 1)
        session_sl_mult = 1.0
        session_tp_mult = 1.0
        session_scale_n = getattr(profile, 'scale_entries', 1)
        session_rr_min = bot.params.min_rr_ratio

        decision = bot.continuous_decider.decide(
            signal.distribution if signal.distribution else None,
            regime, profile, ltf_df,
        )
        bot._last_decision[symbol] = decision
        logger.info(f"[{symbol}] Decisión: vol={decision.suggested_volume_pct:.1f}x conv={decision.conviction:.0%} "
                    f"SLx={decision.sl_width_multiplier:.1f} TPx={decision.tp_width_multiplier:.1f}")

        bot._manage_symbol_position(symbol, current_signal=signal)

        # ── Adaptive Cooldown System (MEJORA 14) ──
        if getattr(bot, '_adaptive_cooldown_enabled', True):
            atr_ratio = regime.atr_ratio if regime else 1.0
            vol_reg = "HIGH" if atr_ratio > 1.5 else "LOW" if atr_ratio < 0.7 else "MEDIUM"
            pattern_name = signal.primary_pattern.type.name if signal.primary_pattern else None
            last_trade_ts = None
            if sym_data.get("last_trade_time"):
                last_trade_ts = sym_data["last_trade_time"].timestamp()

            cd = bot.adaptive_cooldown.evaluate(
                symbol=symbol, pattern=pattern_name,
                atr_ratio=atr_ratio, volatility_regime=vol_reg,
                last_trade_time=last_trade_ts,
            )

            if cd.reason.value:
                for note in cd.notes:
                    logger.info(f"[{symbol}] Cooldown: {note}")

            if cd.active:
                logger.info(f"[{symbol}] Cooldown activo ({cd.remaining_minutes:.0f}min restantes): "
                            f"{cd.reason.value}")
                self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, f"cooldown_{cd.reason.value}" if cd.reason.value else "cooldown", atr_val, pip_value)
                if cd.recommended_volume_mult == 0.0:
                    return

            # Aplicar volumen mínimo y convicción mínima recomendados
            if cd.recommended_volume_mult < 1.0:
                logger.info(f"[{symbol}] Cooldown: volumen ×{cd.recommended_volume_mult:.2f} "
                            f"(stage={cd.recovery_stage.value})")
            bot._cd_volume_mult = cd.recommended_volume_mult
            bot._cd_min_conviction = cd.recommended_min_conviction
        else:
            # Fallback: comportamiento original
            hot_pause_until = 0
            cooldown_bars = getattr(profile, 'cooldown_bars_m1', 0)
            if cooldown_bars > 0:
                last_time = sym_data.get("last_trade_time")
                if last_time:
                    elapsed_min = (datetime.now(timezone.utc).replace(tzinfo=None) - last_time).total_seconds() / 60
                    if elapsed_min < cooldown_bars:
                        remaining = cooldown_bars - elapsed_min
                        logger.info(f"[{symbol}] Cooldown activo ({remaining:.0f}min), saltando entrada")
                        return

        if signal.direction not in ("BUY", "SELL"):
            self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "direccion_hold", atr_val, pip_value)
            return
        base_min_conv = bot.config.get("strategy", {}).get("adaptive", {}).get("min_conviction_to_trade", 0.15)
        adaptive_min_conv = getattr(bot, '_cd_min_conviction', 0.0)
        min_conv = max(base_min_conv, adaptive_min_conv)
        if min_conv > base_min_conv:
            logger.info(f"[{symbol}] Min conviction adaptativo: {min_conv:.0%} (base {base_min_conv:.0%}, "
                        f"extra +{min_conv - base_min_conv:.0%})")
        if conv < min_conv:
            self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "conviccion_baja", atr_val, pip_value)
            logger.info(f"[{symbol}] Convicción {conv:.0%} < {min_conv:.0%}, saltando entrada")
            return
        if not decision.should_trade:
            self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "decision_no_operar", atr_val, pip_value)
            logger.info(f"[{symbol}] Decisión indica no operar, saltando")
            return

        if not is_in_session(now, profile.allowed_sessions):
            self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "fuera_de_sesion", atr_val, pip_value)
            logger.info(f"[{symbol}] Fuera de sesión activa {profile.allowed_sessions}, saltando")
            return

        for avoided in session_profile.avoided_patterns:
            if signal.primary_pattern and avoided in signal.primary_pattern.type.name.upper():
                if conv < 0.5:
                    self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "patron_evitado_en_sesion", atr_val, pip_value)
                    logger.info(f"[{symbol}] Patrón {signal.primary_pattern.type.name} evitado en sesión {session_profile.label}, saltando")
                    return
                logger.info(f"[{symbol}] Patrón {signal.primary_pattern.type.name} evitado en sesión {session_profile.label}, "
                            f"pero convicción {conv:.0%} ≥ 50%, permitiendo")

        breaker_state = bot.circuit_breakers.check_all(
            symbol, ltf_df, bot.news_calendar, now,
            positions_info=bot.mt5.get_positions(symbol) if hasattr(bot.mt5, 'get_positions') else None,
        )
        if breaker_state.any_active():
            bsev = breaker_state.highest_severity()
            logger.warning(f"[{symbol}] Circuit breaker {bsev}: "
                          f"{' | '.join(s.reason for s in [breaker_state.volatility_spike, breaker_state.news_approaching, breaker_state.momentum_against] if s.active)}")
            if bsev == "critical":
                self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, f"circuit_breaker_{bsev}", atr_val, pip_value)
                logger.info(f"[{symbol}] Circuit breaker CRITICAL, saltando entrada")
                return

        open_positions = bot.mt5.get_positions(symbol)
        pyramid_entry = False
        max_pos = getattr(profile, 'max_concurrent_trades', 1)
        if open_positions:
            if len(open_positions) >= max_pos:
                if signal.score >= bot.high_confidence_score:
                    logger.info(f"[{symbol}] Max pos ({len(open_positions)}/{max_pos}) but score {signal.score:.0f} >= high conf, allowing pyramid")
                    pyramid_entry = True
                else:
                    self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, "max_positions_activas", atr_val, pip_value)
                    logger.info(f"[{symbol}] Max positions ({len(open_positions)}/{max_pos}), score {signal.score:.0f} < {bot.high_confidence_score}, skipping")
                    return
            else:
                pyramid_entry = True

        primary = signal.primary_pattern
        is_gap_pattern = primary is not None and (
            primary.type.name.startswith("FVG") or primary.type.name.startswith("VOID_SCALP")
        )
        if is_gap_pattern:
            gap_mid = primary.low + (primary.high - primary.low) * 0.5
            gap_distance = abs(ltf_df["close"].iloc[-1] - gap_mid)
            sl_max_pips = getattr(profile, 'sl_max_pips', None)
            if sl_max_pips is None:
                sl_max_pips = 20.0
            max_dist = sl_max_pips * pip_value * 3
            if gap_distance > max_dist:
                logger.info(f"[{symbol}] Gap distance {gap_distance/pip_value:.0f}p > {sl_max_pips*3:.0f}p, considering LIMIT @ {gap_mid}")

                # Check existing pending orders (same direction)
                pending_same = [(t, inf) for t, inf in bot.pending_orders.items()
                                if inf.get("symbol") == symbol and inf.get("direction") == signal.direction]
                if pending_same:
                    old_entry = pending_same[0][1].get("poi_level")
                    entry_diff = abs(old_entry - gap_mid) if old_entry else 1e9
                    if entry_diff <= pip_value * 5:
                        logger.info(f"[{symbol}] Gap ORDER ya existe @ {old_entry:.5f}, duplicado evitado")
                        return
                    logger.info(f"[{symbol}] Gap entry cambió de {old_entry:.5f} a {gap_mid:.5f}, cancelando {len(pending_same)} pendientes")
                    for t, _ in pending_same:
                        bot.executor.cancel_pending_order(t)
                        bot.pending_orders.pop(t, None)

                # Cancel opposite pending orders
                opposite_dir = "BUY" if signal.direction == "SELL" else "SELL"
                opposite_pending = [(t, inf) for t, inf in bot.pending_orders.items()
                                    if inf.get("symbol") == symbol and inf.get("direction") == opposite_dir]
                if opposite_pending:
                    logger.info(f"[{symbol}] Cancelando {len(opposite_pending)} órdenes {opposite_dir} opuestas antes de gap {signal.direction}")
                    for t, _ in opposite_pending:
                        bot.executor.cancel_pending_order(t)
                        bot.pending_orders.pop(t, None)

                # Check position limit (respect max_concurrent_trades)
                open_positions = bot.mt5.get_positions(symbol)
                max_pos = getattr(profile, 'max_concurrent_trades', 1)
                if open_positions and len(open_positions) >= max_pos:
                    if signal.score < bot.high_confidence_score:
                        logger.info(f"[{symbol}] Gap ORDER saltada: max pos ({len(open_positions)}/{max_pos}) y score {signal.score:.0f} < {bot.high_confidence_score}")
                        return
                    logger.info(f"[{symbol}] Gap ORDER permitida: max pos ({len(open_positions)}/{max_pos}) pero score alto {signal.score:.0f} >= {bot.high_confidence_score}")

                volume = bot._calc_volume(signal, profile, ltf_df, pip_value, symbol)
                if volume > 0:
                    result = bot.executor.place_pending_limit(signal, volume, gap_mid)
                    if result.success:
                        bot.pending_orders[result.order_ticket] = {
                            "symbol": symbol, "direction": signal.direction,
                            "poi_level": gap_mid, "signal_score": signal.score,
                            "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        }
                        sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                return

        from src.utils.helpers import atr
        atr_val = atr(ltf_df, 14).iloc[-1]

        sl_min_pips = getattr(profile, 'sl_min_pips', 5.0)
        spread_pips = 0.0
        symbol_info = bot.mt5.get_symbol_info(symbol)
        if symbol_info:
            spread_points = symbol_info.get("spread", 0)
            spread_pips = spread_points / (10 ** (symbol_info.get("digits", 5) - 1))
            max_spread = bot.config["strategy"].get("params", {}).get("max_spread_pips", 15.0)
            if spread_pips > max_spread:
                logger.warning(f"[{symbol}] Spread {spread_pips:.1f} pips exceeds max, skipping")
                return
            if spread_pips * 2 > sl_min_pips:
                sl_min_pips = spread_pips * 2
            if isinstance(signal.stop_loss, (int, float)) and signal.stop_loss > 0:
                sl_distance = abs(signal.entry_price - signal.stop_loss) / pip_value
                if sl_distance < sl_min_pips:
                    new_sl_dist = sl_min_pips * pip_value
                    if signal.direction == "BUY":
                        adjusted_sl = signal.entry_price - new_sl_dist
                    else:
                        adjusted_sl = signal.entry_price + new_sl_dist
                    signal.stop_loss = adjusted_sl
                    logger.info(f"[{symbol}] SL ajustado de {sl_distance:.1f} a {sl_min_pips:.1f} pips")

        use_route_sl = (
            market_result is not None
            and market_result.route is not None
            and market_result.route.is_valid
            and market_result.stop_loss is not None
        )
        if use_route_sl:
            signal.stop_loss = market_result.stop_loss
            if market_result.suggested_entry is not None:
                signal.entry_price = market_result.suggested_entry
            if market_result.active_tp is not None and market_result.active_tp > 0:
                signal.take_profit = market_result.active_tp
            else:
                rr_min = max(getattr(profile, 'min_rr_ratio', 2.0), 1.5)
                sl_dist = abs(signal.entry_price - signal.stop_loss)
                signal.take_profit = (
                    signal.entry_price + sl_dist * rr_min
                    if signal.direction == "BUY"
                    else signal.entry_price - sl_dist * rr_min
                )
            logger.info(f"[{symbol}] Route SL/TP: entry={signal.entry_price:.5f} SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f}")
        else:
            digits = symbol_info.get("digits", 5) if symbol_info else 5
            base_sl = getattr(bot.risk_manager.config, 'atr_multiplier_sl', 1.5)
            base_tp = getattr(bot.risk_manager.config, 'atr_multiplier_tp', 2.0)
            vol_entry = bot.volatility_scaler.adjust_sl_tp(
                symbol, ltf_df, signal.entry_price, signal.direction,
                signal.stop_loss, signal.take_profit,
                digits=digits, pip=pip_value,
                base_sl_mult=base_sl * session_sl_mult,
                base_tp_mult=base_tp * session_tp_mult,
                sl_min_pips=sl_min_pips,
                sl_max_pips=getattr(profile, 'sl_max_pips', 35.0),
            )
            signal.stop_loss, signal.take_profit = vol_entry[0], vol_entry[1]
            logger.info(f"[{symbol}] {vol_entry[2]['note']}")

        if not bot._positions_cache:
            bot._positions_cache = bot.mt5.get_positions()
        all_positions = bot._positions_cache
        active_dirs = {}
        for p in all_positions:
            sym = p.get("symbol", p.get("_symbol", ""))
            active_dirs[sym] = "BUY" if p.get("type") == "buy" else "SELL"
        corr_confirmed, corr_reason = bot.correlation_engine.confirm_signal(
            symbol, signal.direction.upper(),
            list(bot.symbols.keys()), active_dirs,
        )
        if not corr_confirmed:
            logger.info(f"[{symbol}] Correlación BLOQUEA entrada: {corr_reason}")
            return
        if corr_reason:
            logger.info(f"[{symbol}] Correlación: {corr_reason}")

        # ── Portfolio Risk: pre-check consolidado ──
        if getattr(profile, 'isolated_risk', False):
            port_ok, port_reason, port_vol_mult = True, "isolated_risk", 1.0
            logger.info(f"[{symbol}] Riesgo aislado: saltando portfolio check")
        else:
            bot.portfolio_risk.update_positions({
                p.get("symbol", p.get("_symbol", "")): {
                    "direction": "BUY" if p.get("type") == "buy" else "SELL",
                    "volume": p.get("volume", 0),
                    "entry": p.get("price_open", 0),
                }
                for p in all_positions
            })
            port_ok, port_reason, port_vol_mult = bot.portfolio_risk.pre_check(
                symbol, signal.direction, signal.score, conv,
                list(bot.symbols.keys()),
            )
            if not port_ok:
                logger.info(f"[{symbol}] Portfolio risk BLOQUEA: {port_reason}")
                return
            if port_vol_mult < 1.0:
                logger.info(f"[{symbol}] Portfolio risk ajusta volumen ×{port_vol_mult:.2f}: {port_reason}")

        volume = bot._calc_volume(signal, profile, ltf_df, pip_value, symbol, conviction=conv)
        if port_vol_mult < 1.0:
            volume *= port_vol_mult
            volume = max(0.01, round(volume, 2))
            logger.info(f"[{symbol}] Volumen post-portfolio: {volume} lots")
        if volume <= 0:
            return

        corr_vol_adj, corr_vol_reason = bot.correlation_engine.volume_adjustment(
            symbol, {p["symbol"]: p["volume"] for p in all_positions},
        )
        if corr_vol_adj < 1.0:
            volume *= corr_vol_adj
            logger.info(f"[{symbol}] Vol ajustado por correlación: ×{corr_vol_adj:.2f} ({corr_vol_reason})")

        scale_n = session_scale_n
        min_lot = 0.01
        per_unit = max(round(volume / scale_n / min_lot) * min_lot, min_lot) if scale_n > 1 else volume
        actual_volume = per_unit * scale_n if scale_n > 1 else volume
        if actual_volume > getattr(profile, 'max_volume', 0) > 0:
            actual_volume = min(actual_volume, profile.max_volume)
            per_unit = max(round(actual_volume / scale_n / min_lot) * min_lot, min_lot)

        entry_price = signal.entry_price

        symbol_info_2 = bot.mt5.get_symbol_info(symbol)
        if symbol_info_2:
            if signal.direction == "SELL" and entry_price <= symbol_info_2["bid"]:
                entry_price = symbol_info_2["bid"] + atr_val * 0.5
                logger.info(f"[{symbol}] SELL entry ajustado a {entry_price:.5f} (bid={symbol_info_2['bid']:.5f})")
            elif signal.direction == "BUY" and entry_price >= symbol_info_2["ask"]:
                entry_price = symbol_info_2["ask"] - atr_val * 0.5
                logger.info(f"[{symbol}] BUY entry ajustado a {entry_price:.5f} (ask={symbol_info_2['ask']:.5f})")

        order_decision = bot.order_selector.select(
            direction=signal.direction,
            signal_entry=entry_price,
            current_ask=symbol_info_2["ask"] if symbol_info_2 else entry_price,
            current_bid=symbol_info_2["bid"] if symbol_info_2 else entry_price,
            atr_val=atr_val,
            spread_pips=spread_pips,
            score=signal.score,
            conviction=conv,
            is_gap_pattern=is_gap_pattern,
            regime_trend_alignment=signal.regime_context.trend_alignment if signal.regime_context else "NEUTRAL",
            adx=signal.regime_context.adx_value if signal.regime_context else 0.0,
            min_stop_distance_atr=bot.params.min_stop_distance_atr,
        )
        entry_price = order_decision.entry_price

        conv_old = conv
        conv *= order_decision.confidence_adjustment
        conv = min(1.0, max(0.01, conv))
        if abs(conv - conv_old) > 0.001:
            kelly_ratio = bot.kelly_risk.get_adjusted_volume_mult(conv) / max(
                bot.kelly_risk.get_adjusted_volume_mult(conv_old), 0.001
            )
            volume = max(0.01, round(volume * kelly_ratio, 2))
            logger.info(f"[{symbol}] Order type ajusta convicción {conv_old:.0%} → {conv:.0%}, volumen ×{kelly_ratio:.2f}")

        logger.info(f"[{symbol}] Order type: {order_decision.order_type.value.upper()} @ {entry_price:.5f} — {order_decision.reason}")

        if not bot.hours_controller.can_open_new_position(symbol):
            logger.info(f"[{symbol}] MarketHours restrict: {bot.hours_controller.get_state(symbol).value} → skipping entry")
            self._record_skipped(bot, symbol, signal, conv, regime, session_profile, ltf_df, f"market_hours_{bot.hours_controller.get_state(symbol).value}", atr_val, pip_value)
            return

        if order_decision.order_type == OrderType.MARKET:
            if volume > 0:
                result = bot.executor.place_market_order(signal, volume, price=entry_price)
                if result.success:
                    sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                    logger.info(f"[{symbol}] Market entry: {result.message}")
                else:
                    logger.warning(f"[{symbol}] Market entry failed: {result.message}")
            return

        opposite_dir = "BUY" if signal.direction == "SELL" else "SELL"
        opposite_pending = [(t, info) for t, info in bot.pending_orders.items()
                            if info.get("symbol") == symbol and info.get("direction") == opposite_dir]
        if opposite_pending:
            logger.info(f"[{symbol}] Cancelando {len(opposite_pending)} órdenes {opposite_dir} opuestas antes de {signal.direction}")
            for t, _ in opposite_pending:
                bot.executor.cancel_pending_order(t)
                bot.pending_orders.pop(t, None)

        pending_same = [(t, info) for t, info in bot.pending_orders.items()
                        if info.get("symbol") == symbol and info.get("direction") == signal.direction]
        if pending_same:
            old_entry = pending_same[0][1].get("poi_level")
            entry_diff = abs(old_entry - entry_price) if old_entry else 0
            if entry_diff > atr_val * 0.5:
                logger.info(f"[{symbol}] Entry price cambió de {old_entry:.5f} a {entry_price:.5f}, cancelando {len(pending_same)} pendientes")
                for t, _ in pending_same:
                    bot.executor.cancel_pending_order(t)
                    bot.pending_orders.pop(t, None)
            else:
                logger.info(f"[{symbol}] Ya hay {len(pending_same)} órdenes {signal.direction} activas @ {old_entry:.5f}, saltando duplicado")
                return

        if scale_n > 1 and per_unit >= min_lot:
            zone_id = str(uuid4())
            pend_tickets = []
            ot_label = order_decision.order_type.value.upper()
            logger.info(f"[{symbol}] Scale {ot_label}: {scale_n}x{per_unit} lots @ {entry_price}, zone={zone_id[:8]} — {order_decision.reason}")
            for i in range(scale_n):
                result = bot.executor.place_order(order_decision.order_type, signal, per_unit, entry_price)
                if result.success:
                    pend_tickets.append(result.order_ticket)
                    bot.pending_orders[result.order_ticket] = {
                        "symbol": symbol, "direction": signal.direction,
                        "poi_level": entry_price, "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        "zone_id": zone_id, "volume": per_unit,
                        "sl": signal.stop_loss, "tp": signal.take_profit,
                    }
                else:
                    logger.warning(f"[{symbol}] Pending order {i+1}/{scale_n} failed: {result.message}")
            if pend_tickets:
                bot._pending_batches.setdefault(symbol, {})[zone_id] = {
                    "tickets": pend_tickets,
                    "direction": signal.direction,
                    "entry_price": entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "scale_n": scale_n,
                    "filled_tickets": [],
                }
                sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                logger.info(f"[{symbol}] Pending batch {zone_id[:8]}: {len(pend_tickets)}/{scale_n} placed")
                if pyramid_entry:
                    bot._widen_runners_tp(symbol, signal, entry_price, pip_value)
        elif scale_n <= 1 or per_unit < min_lot:
            result = bot.executor.place_order(order_decision.order_type, signal, volume, entry_price)
            if result.success:
                sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                bot.pending_orders[result.order_ticket] = {
                    "symbol": symbol, "direction": signal.direction,
                    "poi_level": entry_price, "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    "zone_id": None, "volume": volume,
                    "sl": signal.stop_loss, "tp": signal.take_profit,
                }
                ot_label = order_decision.order_type.value.upper()
                logger.info(f"[{symbol}] {ot_label} order placed: {result.message}")
            else:
                logger.warning(f"[{symbol}] Pending order failed: {result.message}")
