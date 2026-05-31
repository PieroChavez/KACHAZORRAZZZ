"""
visualization.py

Funciones utilitarias para visualización de backtesting en Plotly/Streamlit.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from strategies.backtest_engine import Trade, TradeDirection, TradeStatus


def create_equity_curve_chart(equity_data: List[Dict], trades: List[Trade]) -> go.Figure:
    """Crea gráfico de curva de equity con marcadores de trades."""
    if not equity_data:
        return go.Figure()
    
    df_eq = pd.DataFrame(equity_data)
    
    fig = go.Figure()
    
    # Línea de equity
    fig.add_trace(go.Scatter(
        x=df_eq['time'],
        y=df_eq['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#089981', width=2),
        fill='tozeroy',
        fillcolor='rgba(8, 153, 129, 0.1)'
    ))
    
    # Línea de balance (sin trades abiertos)
    fig.add_trace(go.Scatter(
        x=df_eq['time'],
        y=df_eq['balance'],
        mode='lines',
        name='Balance',
        line=dict(color='#2157f3', width=1, dash='dot')
    ))
    
    # Marcadores de trades cerrados
    if trades:
        closed_trades = [t for t in trades if t.exit_time and t.status != TradeStatus.OPEN]
        
        # Trades ganadores
        winners = [t for t in closed_trades if t.pnl > 0]
        if winners:
            fig.add_trace(go.Scatter(
                x=[t.exit_time for t in winners],
                y=[t.exit_price if t.exit_price else 0 for t in winners],
                mode='markers',
                name='✅ Wins',
                marker=dict(symbol='triangle-up', size=10, color='#089981'),
                hovertemplate='<b>%{customdata[0]}</b><br>PnL: $%{customdata[1]:.2f}<br>RR: %{customdata[2]:.2f}<extra></extra>',
                customdata=[[t.id, t.pnl, abs(t.pnl/(t.entry_price-t.sl)) if t.sl != t.entry_price else 0] for t in winners]
            ))
        
        # Trades perdedores
        losers = [t for t in closed_trades if t.pnl <= 0]
        if losers:
            fig.add_trace(go.Scatter(
                x=[t.exit_time for t in losers],
                y=[t.exit_price if t.exit_price else 0 for t in losers],
                mode='markers',
                name='❌ Losses',
                marker=dict(symbol='triangle-down', size=10, color='#F23645'),
                hovertemplate='<b>%{customdata[0]}</b><br>PnL: $%{customdata[1]:.2f}<br>RR: %{customdata[2]:.2f}<extra></extra>',
                customdata=[[t.id, t.pnl, abs(t.pnl/(t.entry_price-t.sl)) if t.sl != t.entry_price else 0] for t in losers]
            ))
    
    fig.update_layout(
        title='📈 Curva de Equity',
        xaxis_title='Fecha',
        yaxis_title='Balance ($)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333')
    )
    
    return fig


def create_drawdown_chart(equity_data: List[Dict]) -> go.Figure:
    """Crea gráfico de drawdown."""
    if not equity_data or len(equity_data) < 2:
        return go.Figure()
    
    df_eq = pd.DataFrame(equity_data)
    
    # Calcular drawdown
    df_eq['peak'] = df_eq['equity'].cummax()
    df_eq['drawdown'] = (df_eq['peak'] - df_eq['equity']) / df_eq['peak'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_eq['time'],
        y=df_eq['drawdown'],
        mode='lines',
        name='Drawdown %',
        line=dict(color='#F23645', width=2),
        fill='tozeroy',
        fillcolor='rgba(242, 54, 69, 0.2)'
    ))
    
    # Línea de drawdown máximo
    max_dd = df_eq['drawdown'].max()
    fig.add_hline(
        y=max_dd,
        line=dict(color='#F23645', width=1, dash='dash'),
        annotation_text=f'Max DD: {max_dd:.2f}%',
        annotation_position='right'
    )
    
    fig.update_layout(
        title='📉 Drawdown',
        xaxis_title='Fecha',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333', range=[0, max(50, max_dd * 1.2)])
    )
    
    return fig


def create_trade_distribution_chart(trades: List[Trade]) -> go.Figure:
    """Crea gráfico de distribución de trades por PnL."""
    if not trades:
        return go.Figure()
    
    closed = [t for t in trades if t.status != TradeStatus.OPEN]
    if not closed:
        return go.Figure()
    
    pnl_values = [t.pnl for t in closed]
    
    fig = go.Figure(data=[
        go.Histogram(
            x=pnl_values,
            nbinsx=30,
            marker_color=['#089981' if x > 0 else '#F23645' for x in pnl_values],
            opacity=0.8,
            name='Distribución PnL'
        )
    ])
    
    # Línea en cero
    fig.add_vline(x=0, line=dict(color='#ffffff', width=2, dash='dot'))
    
    fig.update_layout(
        title='📊 Distribución de PnL por Trade',
        xaxis_title='PnL ($)',
        yaxis_title='Frecuencia',
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333')
    )
    
    return fig


def create_metrics_dashboard(metrics: Dict) -> Dict[str, any]:
    """Prepara métricas para mostrar en Streamlit."""
    if not metrics:
        return {}
    
    # Determinar colores según valores
    def color_metric(value: float, good_threshold: float, bad_threshold: float) -> str:
        if value >= good_threshold:
            return "🟢"
        elif value <= bad_threshold:
            return "🔴"
        return "🟡"
    
    return {
        'win_rate': {
            'value': f"{metrics.get('win_rate', 0):.1f}%",
            'delta': f"{metrics.get('win_rate', 0) - 50:+.1f}%",
            'color': 'normal' if metrics.get('win_rate', 50) >= 50 else 'inverse'
        },
        'profit_factor': {
            'value': f"{metrics.get('profit_factor', 0):.2f}",
            'icon': color_metric(metrics.get('profit_factor', 0), 2.0, 1.0),
            'help': '>2.0 Excelente | >1.5 Bueno | <1.0 Perdedor'
        },
        'max_drawdown': {
            'value': f"{metrics.get('max_drawdown', 0):.1f}%",
            'delta': f"{metrics.get('max_drawdown', 0):+.1f}%",
            'color': 'inverse'  # Menor es mejor
        },
        'total_pnl': {
            'value': f"${metrics.get('total_pnl', 0):,.2f}",
            'delta': f"{metrics.get('total_pnl_percent', 0):+.1f}%",
            'color': 'normal' if metrics.get('total_pnl', 0) >= 0 else 'inverse'
        },
        'expectancy': {
            'value': f"${metrics.get('expectancy', 0):.2f}",
            'help': 'Ganancia esperada por trade'
        },
        'sharpe': {
            'value': f"{metrics.get('sharpe_ratio', 0):.2f}" if metrics.get('sharpe_ratio') else "N/A",
            'help': '>1.0 Bueno | >2.0 Excelente'
        }
    }


def create_price_chart_with_trades(
    df: pd.DataFrame,
    trades: List[Trade],
    smc_results: Optional[Dict] = None,
    title: str = "Precio con Trades"
) -> go.Figure:
    """
    Crea gráfico de velas con marcadores de entrada/salida de trades
    y opcionalmente elementos SMC (OBs, FVGs).
    """
    fig = go.Figure()
    
    # 1. Velas
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Precio",
        increasing_line_color="#089981",
        decreasing_line_color="#F23645",
        opacity=0.9
    ))
    
    # 2. Order Blocks (si se proporcionan resultados SMC)
    if smc_results and smc_results.get('ob_boxes'):
        for ob in smc_results['ob_boxes'][:15]:  # Limitar para rendimiento
            color = ob.get('color', '#1848cc' if ob.get('type') == 'BULLISH_OB' else '#b22833')
            fig.add_vrect(
                x0=ob['left_time'],
                x1=ob.get('right_time', df['time'].iloc[-1]),
                y0=ob['bottom'],
                y1=ob['top'],
                fillcolor=color,
                opacity=0.1,
                line=dict(width=1, color=color, dash="dot"),
                layer="below"
            )
    
    # 3. Fair Value Gaps
    if smc_results and smc_results.get('fvg_boxes'):
        for fvg in smc_results['fvg_boxes'][-20:]:
            color = fvg.get('color', '#00ff68' if fvg.get('bias', 1) == 1 else '#ff0008')
            fig.add_vrect(
                x0=fvg['left_time'],
                x1=fvg.get('right_time', df['time'].iloc[-1]),
                y0=fvg['bottom'],
                y1=fvg['top'],
                fillcolor=color,
                opacity=0.08,
                line=dict(width=1, color=color, dash="dash"),
                layer="below"
            )
    
    # 4. Marcadores de entrada de trades
    if trades:
        entries = [t for t in trades if t.entry_time]
        
        # Entradas LONG
        long_entries = [t for t in entries if t.direction == TradeDirection.LONG]
        if long_entries:
            fig.add_trace(go.Scatter(
                x=[t.entry_time for t in long_entries],
                y=[t.entry_price for t in long_entries],
                mode='markers+text',
                name='🟢 Entry LONG',
                marker=dict(symbol='arrow-up', size=12, color='#089981'),
                text=['🟢' for _ in long_entries],
                textposition='top center',
                hovertemplate='<b>%{customdata[0]}</b><br>Entry: %{y:.3f}<br>SL: %{customdata[1]:.3f}<br>TP: %{customdata[2]:.3f}<extra></extra>',
                customdata=[[t.id, t.sl, t.tp] for t in long_entries]
            ))
        
        # Entradas SHORT
        short_entries = [t for t in entries if t.direction == TradeDirection.SHORT]
        if short_entries:
            fig.add_trace(go.Scatter(
                x=[t.entry_time for t in short_entries],
                y=[t.entry_price for t in short_entries],
                mode='markers+text',
                name='🔴 Entry SHORT',
                marker=dict(symbol='arrow-down', size=12, color='#F23645'),
                text=['🔴' for _ in short_entries],
                textposition='bottom center',
                hovertemplate='<b>%{customdata[0]}</b><br>Entry: %{y:.3f}<br>SL: %{customdata[1]:.3f}<br>TP: %{customdata[2]:.3f}<extra></extra>',
                customdata=[[t.id, t.sl, t.tp] for t in short_entries]
            ))
        
        # Salidas con PnL
        closed = [t for t in trades if t.exit_time and t.status != TradeStatus.OPEN]
        for t in closed:
            color = '#089981' if t.pnl > 0 else '#F23645'
            symbol = 'circle' if t.pnl > 0 else 'x'
            pnl_text = f"+${t.pnl:.0f}" if t.pnl > 0 else f"${t.pnl:.0f}"
            
            fig.add_annotation(
                x=t.exit_time,
                y=t.exit_price,
                text=pnl_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                bgcolor="rgba(30,30,30,0.9)",
                bordercolor=color,
                borderwidth=1,
                font=dict(size=8, color="white"),
                yanchor="bottom" if t.direction == TradeDirection.LONG else "top",
                yshift=10 if t.direction == TradeDirection.LONG else -10
            )
    
    # Layout
    fig.update_layout(
        title=title,
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333'),
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    return fig


def create_multi_tf_correlation_chart(trades: List[Trade]) -> go.Figure:
    """Muestra correlación de rendimiento por timeframe."""
    if not trades:
        return go.Figure()
    
    # Agrupar por timeframe
    tf_stats = {}
    for t in trades:
        if t.status == TradeStatus.OPEN or not t.timeframe:
            continue
        tf = t.timeframe
        if tf not in tf_stats:
            tf_stats[tf] = {'wins': 0, 'losses': 0, 'pnl': 0, 'count': 0}
        tf_stats[tf]['count'] += 1
        tf_stats[tf]['pnl'] += t.pnl
        if t.pnl > 0:
            tf_stats[tf]['wins'] += 1
        else:
            tf_stats[tf]['losses'] += 1
    
    # Preparar datos
    data = []
    for tf, stats in tf_stats.items():
        win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        data.append({
            'Timeframe': tf,
            'Trades': stats['count'],
            'Win Rate': win_rate,
            'PnL': stats['pnl'],
            'Avg PnL': stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        })
    
    df_tf = pd.DataFrame(data)
    
    # Gráfico de barras
    fig = px.bar(
        df_tf,
        x='Timeframe',
        y='Win Rate',
        color='PnL',
        color_continuous_scale=['#F23645', '#FFA500', '#089981'],
        title='🔗 Rendimiento por Timeframe',
        labels={'Win Rate': 'Win Rate (%)', 'PnL': 'PnL Total ($)'},
        text_auto='.1f'
    )
    
    fig.update_layout(
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='#ffffff'),
        coloraxis_colorbar=dict(title='PnL ($)'),
        xaxis=dict(gridcolor='#333333'),
        yaxis=dict(gridcolor='#333333', range=[0, 100])
    )
    
    return fig