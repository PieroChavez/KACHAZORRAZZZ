"""
backtest_engine.py

Motor de Backtesting para Smart Money Concepts
Soporte multi-timeframe y gestión de trades simulados.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from loguru import logger
import json
from datetime import datetime

from strategies.kachazorraz import SmartMoneyConcepts, detect_smc


# ============================================================
# ENUMS Y CONSTANTES
# ============================================================

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"
    TP_HIT = "TP_HIT"


@dataclass
class Trade:
    """Representa una operación simulada."""
    id: str
    direction: TradeDirection
    entry_time: any
    entry_price: float
    exit_time: Optional[any] = None
    exit_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    size: float = 1.0  # Lotes o unidades
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_percent: float = 0.0
    reason: str = ""
    tags: List[str] = field(default_factory=list)  # ej: ["OB_BULLISH", "FVG_CONFLUENCE"]
    timeframe: str = ""
    higher_tf_trend: str = ""  # Tendencia del timeframe superior


@dataclass
class BacktestConfig:
    """Configuración del backtest."""
    symbol: str
    timeframes: List[str]  # ej: ["D1", "H1", "M15"]
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.02  # 2% del balance
    max_trades: int = 100
    use_confluence: bool = True  # Requiere múltiples señales para entrar
    min_rr: float = 1.5  # Mínimo Risk/Reward para entrar
    sl_method: str = "ob_opposite"  # "ob_opposite", "fixed_pips", "atr"
    tp_method: str = "rr"  # "rr", "structure", "fixed_pips"
    atr_period: int = 14
    fixed_sl_pips: float = 50
    fixed_tp_pips: float = 100
    commission: float = 0.0  # Comisión por trade
    spread: float = 0.0  # Spread simulado


# ============================================================
# MOTOR DE BACKTESTING
# ============================================================

class BacktestEngine:
    """
    Motor principal de backtesting con análisis multi-timeframe.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_balance = config.initial_balance
        self.open_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Cache de datos por timeframe
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.smc_cache: Dict[str, Dict] = {}
        
        logger.info(f"BacktestEngine iniciado para {config.symbol}")

    def load_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Carga y valida datos para un timeframe."""
        required_cols = ['open', 'high', 'low', 'close', 'time']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Faltan columnas en datos de {timeframe}")
            return False
        
        # Asegurar formato de tiempo
        if df['time'].dtype in ['int64', 'float64']:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Ordenar por tiempo
        df = df.sort_values('time').reset_index(drop=True)
        
        self.data_cache[timeframe] = df
        logger.info(f"Datos cargados: {timeframe} - {len(df)} velas")
        return True

    def analyze_timeframes(self) -> Dict[str, Dict]:
        """
        Analiza todos los timeframes configurados con SMC.
        Retorna diccionario con resultados por timeframe.
        """
        results = {}
        
        # Ordenar timeframes de mayor a menor para análisis jerárquico
        tf_order = self._sort_timeframes_descending(self.config.timeframes)
        
        for tf in tf_order:
            if tf not in self.data_cache:
                logger.warning(f"Sin datos para timeframe: {tf}")
                continue
            
            df = self.data_cache[tf]
            
            # Configurar detector SMC según timeframe
            smc_params = self._get_smc_params_for_timeframe(tf)
            
            # Ejecutar detección
            smc_results = detect_smc(df, **smc_params)
            self.smc_cache[tf] = smc_results
            
            results[tf] = {
                'data': df,
                'smc': smc_results,
                'trend': smc_results.get('trend', {}).get('swing_bias', 0)
            }
            
            logger.info(f"SMC analizado: {tf} | Pivots: {smc_results['stats']['total_pivots']} | BOS: {smc_results['stats']['total_bos']}")
        
        return results

    def _sort_timeframes_descending(self, timeframes: List[str]) -> List[str]:
        """Ordena timeframes de mayor a menor periodicidad."""
        tf_order = {
            'D1': 7, 'W1': 7, 'H4': 4, 'H3': 3, 'H2': 2, 'H1': 1,
            'M30': 0.5, 'M15': 0.25, 'M5': 0.083, 'M2': 0.033, 'M1': 0.016
        }
        return sorted(timeframes, key=lambda x: tf_order.get(x, 0), reverse=True)

    def _get_smc_params_for_timeframe(self, timeframe: str) -> Dict:
        """Configura parámetros SMC según el timeframe."""
        # Parámetros base
        params = {
            'show_structure': True,
            'show_internals': True,
            'show_swing_order_blocks': True,
            'show_internal_order_blocks': True,
            'show_fair_value_gaps': True,
            'show_equal_highs_lows': True,
            'swings_length': 50,
            'equal_highs_lows_length': 3,
            'equal_highs_lows_threshold': 0.1,
        }
        
        # Ajustar según timeframe
        if timeframe in ['D1', 'W1']:
            params['swings_length'] = 30  # Más sensible para timeframe mayor
            params['show_internal_order_blocks'] = False  # Solo swing OBs en D1
        elif timeframe in ['H4', 'H3', 'H1']:
            params['swings_length'] = 40
        elif timeframe in ['M15', 'M5', 'M2']:
            params['swings_length'] = 20  # Más pivotes para entries precisos
            params['equal_highs_lows_threshold'] = 0.05  # Más estricto
        
        return params

    def generate_signals(self, analysis_results: Dict) -> List[Dict]:
        """
        Genera señales de entrada basadas en confluencia multi-timeframe.
        """
        signals = []
        
        # Obtener timeframe de entrada (el más pequeño)
        entry_tf = self._sort_timeframes_descending(self.config.timeframes)[-1]
        higher_tfs = self._sort_timeframes_descending(self.config.timeframes)[:-1]
        
        if entry_tf not in analysis_results:
            return signals
        
        entry_data = analysis_results[entry_tf]
        entry_df = entry_data['data']
        entry_smc = entry_data['smc']
        
        # Obtener tendencia de timeframe superior para filtro direccional
        higher_trend = 0
        if higher_tfs and higher_tfs[0] in analysis_results:
            higher_trend = analysis_results[higher_tfs[0]].get('trend', 0)
        
        # Iterar sobre Order Blocks del timeframe de entrada
        for ob in entry_smc.get('order_blocks', []):
            if getattr(ob, 'mitigated', False):
                continue  # Saltar OBs ya mitigados
            
            signal = self._evaluate_ob_confluence(
                ob=ob,
                entry_df=entry_df,
                entry_smc=entry_smc,
                analysis_results=analysis_results,
                higher_trend=higher_trend,
                entry_timeframe=entry_tf
            )
            
            if signal and signal['valid']:
                signals.append(signal)
        
        logger.info(f"Señales generadas: {len(signals)}")
        return signals

    def _evaluate_ob_confluence(
        self,
        ob,
        entry_df: pd.DataFrame,
        entry_smc: Dict,
        analysis_results: Dict,
        higher_trend: int,
        entry_timeframe: str
    ) -> Optional[Dict]:
        """
        Evalúa confluencia para un Order Block específico.
        Retorna señal válida o None.
        """
        ob_idx = None
        # Buscar índice del OB en el DataFrame
        for idx, row in entry_df.iterrows():
            if row['time'] == ob.bar_time:
                ob_idx = idx
                break
        
        if ob_idx is None or ob_idx >= len(entry_df) - 5:
            return None  # OB muy reciente, no hay datos para confirmar
        
        # Score de confluencia
        score = 0
        factors = []
        
        # 1. Dirección del OB vs tendencia superior
        if higher_trend != 0:
            if (ob.bias == higher_trend):
                score += 2
                factors.append("TREND_ALIGN")
            else:
                score -= 1  # Penalizar contra-tendencia
                factors.append("TREND_COUNTER")
        
        # 2. Presencia de FVG en la misma zona
        fvgs = entry_smc.get('fair_value_gaps', [])
        fvg_confluence = any(
            abs(fvg.bottom - (ob.bar_high + ob.bar_low)/2) < (ob.bar_high - ob.bar_low) * 0.5
            for fvg in fvgs if fvg.bias == ob.bias
        )
        if fvg_confluence:
            score += 2
            factors.append("FVG_CONFLUENCE")
        
        # 3. Equal Highs/Lows cercanos (liquidez)
        eqls = entry_smc.get('equal_highs_lows', [])
        liquidity_near = any(
            abs(eql['price'] - (ob.bar_high if ob.bias == -1 else ob.bar_low)) < (ob.bar_high - ob.bar_low) * 2
            for eql in eqls
        )
        if liquidity_near:
            score += 1
            factors.append("LIQUIDITY_ZONE")
        
        # 4. BOS reciente confirmando dirección
        bos_list = entry_smc.get('bos_list', [])
        recent_bos = any(
            bos['index'] > ob_idx - 10 and 
            ('BULLISH' in bos['type'] if ob.bias == 1 else 'BEARISH' in bos['type'])
            for bos in bos_list
        )
        if recent_bos:
            score += 2
            factors.append("BOS_CONFIRMATION")
        
        # 5. Precio actual cerca del OB (oportunidad de entrada)
        current_price = entry_df['close'].iloc[-1]
        ob_mid = (ob.bar_high + ob.bar_low) / 2
        ob_range = ob.bar_high - ob.bar_low
        price_near_ob = abs(current_price - ob_mid) < ob_range * 3
        
        if not price_near_ob:
            return None  # Precio muy lejos del OB, no es entrada válida
        
        # Validar score mínimo
        min_score = 3 if self.config.use_confluence else 1
        if score < min_score:
            return None
        
        # Calcular niveles de entrada, SL y TP
        entry_price = ob.bar_low if ob.bias == 1 else ob.bar_high  # Entrada en límite del OB
        
        # Stop Loss: lado opuesto del OB + buffer
        sl_buffer = ob_range * 0.3
        sl_price = ob.bar_high + sl_buffer if ob.bias == 1 else ob.bar_low - sl_buffer
        
        # Take Profit: basado en R:R mínimo
        risk = abs(entry_price - sl_price)
        tp_price = entry_price + risk * self.config.min_rr if ob.bias == 1 else entry_price - risk * self.config.min_rr
        
        signal = {
            'valid': True,
            'direction': TradeDirection.LONG if ob.bias == 1 else TradeDirection.SHORT,
            'entry_time': entry_df['time'].iloc[-1],
            'entry_price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'ob': ob,
            'score': score,
            'factors': factors,
            'timeframe': entry_timeframe,
            'higher_tf_trend': 'BULLISH' if higher_trend == 1 else 'BEARISH' if higher_trend == -1 else 'NEUTRAL',
            'tags': factors
        }
        
        return signal

    def execute_backtest(self, signals: List[Dict]) -> List[Trade]:
        """
        Ejecuta el backtest simulando las señales generadas.
        """
        self.trades = []
        self.equity_curve = []
        self.current_balance = self.config.initial_balance
        self.open_trades = []
        self.trade_counter = 0
        
        # Obtener datos del timeframe de entrada para simulación
        entry_tf = self._sort_timeframes_descending(self.config.timeframes)[-1]
        if entry_tf not in self.data_cache:
            logger.error(f"Sin datos para ejecutar backtest en {entry_tf}")
            return []
        
        df = self.data_cache[entry_tf]
        
        # Simular barra por barra desde la primera señal
        if not signals:
            logger.warning("No hay señales para ejecutar backtest")
            return []
        
        start_idx = min(sig.get('entry_time') for sig in signals if isinstance(sig.get('entry_time'), int))
        start_idx = df[df['time'] >= pd.to_datetime(start_idx, unit='s') if isinstance(start_idx, (int, float)) else start_idx].index[0] if isinstance(start_idx, (int, float)) else 0
        
        for idx in range(start_idx, len(df)):
            current_bar = df.iloc[idx]
            current_time = current_bar['time']
            current_price = current_bar['close']
            
            # 1. Verificar nuevas señales para entrar
            for signal in signals:
                if signal.get('processed'):
                    continue
                
                signal_time = signal['entry_time']
                # Convertir a comparable
                if isinstance(signal_time, (int, float)):
                    signal_time = pd.to_datetime(signal_time, unit='s')
                
                if signal_time <= current_time:
                    self._open_trade(signal, current_price, entry_tf)
                    signal['processed'] = True
            
            # 2. Gestionar trades abiertos (SL, TP, cierre)
            self._manage_open_trades(current_bar, current_price)
            
            # 3. Registrar equity curve cada N velas
            if idx % 10 == 0:
                self.equity_curve.append({
                    'time': current_time,
                    'balance': self.current_balance,
                    'equity': self._calculate_current_equity(current_price),
                    'open_trades': len(self.open_trades)
                })
        
        # Cerrar trades pendientes al final
        for trade in self.open_trades[:]:
            self._close_trade(trade, df.iloc[-1]['close'], df.iloc[-1]['time'], "END_OF_BACKTEST")
        
        logger.info(f"Backtest completado: {len(self.trades)} trades, Balance final: ${self.current_balance:.2f}")
        return self.trades

    def _open_trade(self, signal: Dict, current_price: float, timeframe: str):
        """Abre una nueva operación simulada."""
        self.trade_counter += 1
        
        # Calcular tamaño de posición basado en riesgo
        risk_amount = self.current_balance * self.config.risk_per_trade
        risk_pips = abs(signal['entry_price'] - signal['sl'])
        position_size = risk_amount / risk_pips if risk_pips > 0 else 1.0
        
        trade = Trade(
            id=f"T{self.trade_counter:04d}",
            direction=signal['direction'],
            entry_time=signal['entry_time'],
            entry_price=signal['entry_price'],
            sl=signal['sl'],
            tp=signal['tp'],
            size=min(position_size, 10.0),  # Límite máximo de tamaño
            timeframe=timeframe,
            higher_tf_trend=signal.get('higher_tf_trend', 'NEUTRAL'),
            tags=signal.get('factors', [])
        )
        
        self.open_trades.append(trade)
        self.trades.append(trade)
        
        # Deducir comisión/spread al entrar
        self.current_balance -= self.config.commission + self.config.spread * trade.size
        
        logger.info(f"🟢 Trade abierto: {trade.id} | {signal['direction'].value} @ {signal['entry_price']:.3f}")

    def _manage_open_trades(self, current_bar: pd.Series, current_price: float):
        """Gestiona trades abiertos: verifica SL, TP y condiciones de cierre."""
        for trade in self.open_trades[:]:  # Copia para poder remover
            if trade.status != TradeStatus.OPEN:
                continue
            
            # Verificar Stop Loss
            if trade.direction == TradeDirection.LONG and current_price <= trade.sl:
                self._close_trade(trade, trade.sl, current_bar['time'], "STOP_LOSS")
                continue
            elif trade.direction == TradeDirection.SHORT and current_price >= trade.sl:
                self._close_trade(trade, trade.sl, current_bar['time'], "STOP_LOSS")
                continue
            
            # Verificar Take Profit
            if trade.direction == TradeDirection.LONG and current_price >= trade.tp:
                self._close_trade(trade, trade.tp, current_bar['time'], "TAKE_PROFIT")
                continue
            elif trade.direction == TradeDirection.SHORT and current_price <= trade.tp:
                self._close_trade(trade, trade.tp, current_bar['time'], "TAKE_PROFIT")
                continue
            
            # Cierre por tiempo máximo (opcional)
            # if trade.entry_time and (current_bar['time'] - trade.entry_time).total_seconds() > MAX_HOLD_SECONDS:
            #     self._close_trade(trade, current_price, current_bar['time'], "TIME_EXIT")

    def _close_trade(self, trade: Trade, exit_price: float, exit_time: any, reason: str):
        """Cierra una operación y calcula PnL."""
        # Calcular PnL
        if trade.direction == TradeDirection.LONG:
            pnl = (exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - exit_price) * trade.size
        
        # Aplicar comisión/spread al salir
        pnl -= self.config.commission + self.config.spread * trade.size
        
        pnl_percent = (pnl / (self.current_balance + pnl)) * 100 if (self.current_balance + pnl) != 0 else 0
        
        # Actualizar trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.status = TradeStatus.TP_HIT if reason == "TAKE_PROFIT" else TradeStatus.STOPPED if reason == "STOP_LOSS" else TradeStatus.CLOSED
        trade.reason = reason
        
        # Actualizar balance
        self.current_balance += pnl
        
        # Remover de abiertos
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        
        status_icon = "✅" if pnl > 0 else "❌"
        logger.info(f"{status_icon} Trade cerrado: {trade.id} | PnL: ${pnl:.2f} ({pnl_percent:+.2f}%) | {reason}")

    def _calculate_current_equity(self, current_price: float) -> float:
        """Calcula equity actual incluyendo trades abiertos."""
        equity = self.current_balance
        for trade in self.open_trades:
            if trade.direction == TradeDirection.LONG:
                unrealized = (current_price - trade.entry_price) * trade.size
            else:
                unrealized = (trade.entry_price - current_price) * trade.size
            equity += unrealized
        return equity

    def get_metrics(self) -> Dict:
        """Calcula métricas de rendimiento del backtest."""
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades if t.status != TradeStatus.OPEN]
        if not closed_trades:
            return {}
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_values = [ec['equity'] for ec in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for eq in equity_values:
                if eq > peak:
                    peak = eq
                drawdown = (peak - eq) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # Expectancy
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss)) if avg_loss else avg_win
        
        return {
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / self.config.initial_balance) * 100,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': abs(avg_loss),
            'avg_rr': np.mean([abs(t.pnl / (t.entry_price - t.sl)) if t.sl != t.entry_price else 0 for t in closed_trades]),
            'sharpe_ratio': self._calculate_sharpe(equity_values) if len(equity_values) > 10 else None,
            'final_balance': self.current_balance,
            'initial_balance': self.config.initial_balance
        }

    def _calculate_sharpe(self, equity_values: List[float], risk_free_rate: float = 0.0) -> Optional[float]:
        """Calcula ratio de Sharpe simplificado."""
        if len(equity_values) < 2:
            return None
        
        returns = np.diff(equity_values) / equity_values[:-1]
        if np.std(returns) == 0:
            return None
        
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns)
        return sharpe * np.sqrt(252)  # Anualizado (asumiendo datos diarios)

    def export_results(self, filepath: str, format: str = 'json') -> str:
        """Exporta resultados del backtest a archivo."""
        metrics = self.get_metrics()
        
        export_data = {
            'config': {
                'symbol': self.config.symbol,
                'timeframes': self.config.timeframes,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_balance': self.config.initial_balance,
                'risk_per_trade': self.config.risk_per_trade
            },
            'metrics': metrics,
            'trades': [
                {
                    'id': t.id,
                    'direction': t.direction.value,
                    'entry_time': str(t.entry_time),
                    'entry_price': t.entry_price,
                    'exit_time': str(t.exit_time) if t.exit_time else None,
                    'exit_price': t.exit_price,
                    'sl': t.sl,
                    'tp': t.tp,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'status': t.status.value,
                    'reason': t.reason,
                    'tags': t.tags,
                    'timeframe': t.timeframe,
                    'higher_tf_trend': t.higher_tf_trend
                }
                for t in self.trades
            ],
            'equity_curve': self.equity_curve[-1000:],  # Limitar para no saturar
            'timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == 'csv':
            # Exportar trades a CSV
            import csv
            with open(filepath, 'w', newline='') as f:
                if export_data['trades']:
                    writer = csv.DictWriter(f, fieldnames=export_data['trades'][0].keys())
                    writer.writeheader()
                    writer.writerows(export_data['trades'])
        
        logger.info(f"Resultados exportados a: {filepath}")
        return filepath
