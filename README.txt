# SMC Scoring Trading Bot

Bot de trading algorítmico para MetaTrader 5 basado en Smart Money Concepts (SMC) con detección de regímenes, memoria de mercado, perfiles de sesión y meta-learning.

## Arquitectura del Pipeline

```
Vela 5min → Evaluación → Contexto → Patrones → Scoring → Convicción → Decisión → Ejecución
                ↑                                                                  │
                └────────── 1min check (BE/TP) ←─── Posiciones abiertas ←──────────┘
```

---

## 1. Patrones Detectados (24 tipos)

### SMC Core
| Patrón | Direcciones | Significado |
|--------|------------|-------------|
| **FVG** (Fair Value Gap) | `FVG_BULLISH` / `FVG_BEARISH` | Gap de 3 velas donde el cuerpo no se solapa. Entrada al 50% del gap. Se invalida si >50% mitigado. |
| **OB** (Order Block) | `OB_BULLISH` / `OB_BEARISH` | Última vela contraria antes de un movimiento direccional. Cuerpo ≥ 0.8 ATR. |
| **BREAKER** | `BREAKER_BULLISH` / `BREAKER_BEARISH` | Precio rompe un swing alto/bajo y cierra de vuelta. Se invalida tras 4+ toques. |
| **SWEEP** (Liquidity Sweep) | `SWEEP_BULLISH` / `SWEEP_BEARISH` | Precio barre un swing previo y cierra de vuelta adentro. |
| **CYCLE** (Ciclo SMC) | `CYCLE_BULLISH` / `CYCLE_BEARISH` | Secuencia de 3 pasos: desliz → equidad → eslabón con cuerpo > 0.7 ATR. |

### Wyckoff
| Patrón | Significado |
|--------|-------------|
| **SPRING** | Ruido: precio rompe soporte y cierra arriba (Fase C). |
| **UTAD** | Señal de debilidad: precio rompe resistencia y cierra abajo (Fase C). |
| **SOS** (Sign of Strength) | Después de un Spring, cierre sobre resistencia con volumen (Fase D). |
| **SOW** (Sign of Weakness) | Después de UTAD, cierre bajo soporte con volumen (Fase D LPS/LPSY). |

### Estructura y Escalpado
| Patrón | Significado |
|--------|-------------|
| **VOID_SCALP** | Escalpado agresivo en el vacío entre desliz y equidad. Tamaño mínimo 0.15 ATR. |
| **SEQUENCE_123** | n velas consecutivas en misma dirección (default 3). |
| **BOS_ZONE_RETEST** | Zona de indecisión seguida de breakout con volumen + retest. |
| **PRICE_ESTABLISHMENT** | Nivel tocado ≥ 3 veces en 20 velas con rechazo. |
| **SUB_FRACTALS** | Conteo de puntos alternantes: 3er movimiento "listo" tras 2 fractales. |

### Conceptos de Curso
| Patrón | Significado |
|--------|-------------|
| **INTERVAL_POINT** | Vela de indecisión con mechas ≥ 2× el cuerpo. Límite de precio institucional. |
| **PRICE_INTERACTION** | Precio toca un swing previo con precisión 0.15 ATR y es absorbido. |
| **HARMONIC_CYCLE** | Retroceso al 50% del último swing completo. |
| **PRESSURE_ZONE** | Falla 1-3 veces en un nivel, consolidación apretada, luego expande. |

### Detecciones Auxiliares
| Detección | Significado |
|-----------|-------------|
| **TRB Manipulation** | Falso breakout + manipulación + desplazamiento + retest. |
| **VSA** (Volume Spread Analysis) | 6 señales: confirmación, clímax, absorción, pullback bajo volumen, divergencia, no-demanda/oferta. |
| **Killzone** | Sesión activa que valida el patrón (Asian, London, NY, Close). |

### Penalizaciones e Invalidaciones
- `body_close_invalid`: Spring/UTAD cerró con cuerpo fuera del rango (breakout real, no spring).
- `no_sweep_detected`: No hay barrido de liquidez en la dirección del trade.
- `fvg_burned_over_50`: FVG mitigado >50%.
- `ob_mitigated_penalty`: Precio violó el límite del OB.
- `breaker_mitigated_penalty`: Precio retomó el nivel del breaker.
- `breaker_3_touch_limit`: Breaker tocado ≥ 4 veces (inválido).
- `lps_mitigated_penalty`: Precio rompió el nivel LPS/LPSY.

---

## 2. Regímenes de Mercado (6 tipos)

Detectados por `RegimeDetector` usando ADX, ratio ATR, pendiente EMA y estructura de swings.

| Régimen | Detección | Multiplicadores de patrón | SL/TP | Volumen |
|---------|-----------|--------------------------|-------|---------|
| **STRONG_TREND_BULLISH** | ADX ≥ 30, fuerza ≥ 0.6, EMA > 0 | FVG/BOS 1.8×, Wyckoff 0.6× | SL 1.0×, TP 1.3× | +20% |
| **STRONG_TREND_BEARISH** | ADX ≥ 30, fuerza ≥ 0.6, EMA < 0 | FVG/BOS 1.8×, Wyckoff 0.6× | SL 1.0×, TP 1.3× | +20% |
| **RANGING** | ADX 20-30, ATR ratio 0.7-1.3 | Sweeps/Wyckoff 1.8×, FVG 0.4× | SL 0.9×, TP 0.7× | -30% |
| **HIGH_VOLATILITY** | ATR ratio > 1.5, ADX < 25 | FVG 1.3×, Wyckoff 0.5× | SL 1.3×, TP 1.5× | -40% |
| **LOW_VOLATILITY** | ATR ratio < 0.7 | OB/Sweeps 1.7× | SL 0.8×, TP 0.9× | Normal |
| **TRANSITION** | Catch-all | Breakers/Wyckoff 1.6× | SL 0.9×, TP 1.2× | Normal |

La confianza del régimen se deriva de ADX/60, escalado por ratio ATR y un contador de estabilidad (requiere 3 detecciones consecutivas del mismo régimen para confianza completa).

---

## 3. Sesiones de Trading (7 tipos)

Perfiladas por `SessionProfiler` con patrones preferidos/evitados y ajustes de volatilidad.

| Sesión | UTC | Patrones Preferidos | Patrones Evitados | Vol. | Volumen |
|--------|-----|---------------------|-------------------|------|---------|
| **ASIAN** | 0-7 | OB, SWEEP, PRESSURE_ZONE | VOID_SCALP, BREAKER, WYCKOFF | ~0.3% | -50% |
| **LONDON_OPEN** | 7-9 | FVG, BREAKER, VOID_SCALP | WYCKOFF, PRESSURE_ZONE | ~0.8% | +20% (peak 8-9) |
| **LONDON_MID** | 9-12 | OB, SWEEP, CYCLE | VOID_SCALP, BREAKER | ~0.5% | Normal |
| **NY_OPEN** | 12-14 | FVG, BREAKER, BOS_ZONE | WYCKOFF, INTERVAL_POINT | ~1.0% | +20% (peak 13-14) |
| **LONDON_NY_OVERLAP** | 12-16 | FVG, BOS_ZONE, CYCLE, BREAKER | Ninguno | ~1.2% | +20% (peak 12-14) |
| **NY_AFTERNOON** | 16-21 | OB, SWEEP, PRESSURE_ZONE | FVG, VOID_SCALP | ~0.6% | -50% (weak 18-20) |
| **CLOSE** | 21-24 | Ninguno | FVG, VOID_SCALP, BREAKER, BOS_ZONE | ~0.2% | -50% |

Los patrones evitados por sesión **se permiten si la convicción ≥ 50%** (override por alta convicción).

---

## 4. Cálculo de Convicción

La convicción (0-100%) se calcula en `DistributionalScorer._compute_conviction()`:

```
convicción = 0.3 × magnitud_score + 0.3 × estabilidad + 0.3 × convergencia + 0.1 × confianza_régimen
```

**Componentes:**
- **magnitud_score**: `min(1.0, |score_neto| / 100)` — qué tan fuerte es el score absoluto
- **estabilidad**: `1 / (1 + σ)` — baja desviación estándar entre timeframes = alta estabilidad
- **convergencia**: fracción de scores alineados en misma dirección entre HTF/MID/LTF (0.0 a 1.0)
- **confianza_régimen**: del RegimeDetector (0.0 a 1.0)

**Ajustes post-cálculo:**
- Multiplicador de patrón por régimen: `convicción × (0.7 + 0.3 × multiplier)`
- Desalineación multi-TF: convicción × 0.7
- Market memory: soporte cercano → BUY +1.2×, SELL 0.5×; resistencia cercana → SELL +1.2×, BUY 0.5×; rango → 0.8×
- Mínimo 30% para operar

---

## 5. Break Even, Trailing y Expansión de TP

### Break Even (BE)
- Se activa cuando el precio alcanza **≥ 30% de la distancia al TP original**
- Mueve el SL al precio de entrada + 0.05 pips de buffer
- `state["be_activated"] = True`

### Trailing Stop
- Solo después de BE activado
- Distancia = ATR(14) × 1.5
- Se actualiza en cada evaluación de 5min
- BUY: `candidato_SL = precio_actual - trail_ATR` (si es mayor que SL actual)
- SELL: `candidato_SL = precio_actual + trail_ATR` (si es menor que SL actual)

### Expansión de TP
- Solo después de BE activado y precio alcanza **≥ 75% de la distancia al TP original**
- Multiplica la distancia original por **1.5×**
- `state["tp_expanded"] = True`
- Permite que los runners sigan la tendencia más allá del TP original

### Scale-Out (Cierres Parciales)
- Cuando el P&L promedio del batch alcanza el `scale_close_at_tp_pct` (default 50% del TP)
- Cierra el primer 50% de los tickets del batch
- Los tickets restantes se convierten en "runners"
- Si hay pyramiding, los runners existentes expanden su TP para igualar el nuevo batch

---

## 6. Cálculo de Volumen

El volumen pasa por 7 etapas:

### Etapa 1: Riesgo base
```
riesgo_pct = profile.risk_per_trade_pct / 100
```

### Etapa 2: Ajuste por convicción
Según `CONVICTION_VOLUME_MAP`:
- Convicción 30% → 0.4×
- Convicción 50% → 0.8×
- Convicción 60% → 1.0× (base)
- Convicción 80% → 1.5×
- Convicción 100% → 3.0× (máximo)

### Etapa 3: Ajuste por régimen
- HIGH_VOLATILITY: ×0.6
- STRONG_TREND: ×1.2
- RANGING: ×0.7
- Alta dispersión (σ > 50% de μ): ×0.7

### Etapa 4: FixedRiskManager
```
volumen = riesgo_usd / (pips_SL × valor_pip)
riesgo_usd = balance × fracción_riesgo
```
La fracción de riesgo se ajusta por convicción:
- Convicción 50% → 1.0× base
- Convicción 80% → 2.0×
- Convicción 100% → 3.0×

### Etapa 5: Cap por perfil
```
volumen = min(volumen, profile.max_volume)
```

### Etapa 6: Ajuste por sesión
- Sesión peak: ×1.2
- Sesión débil: ×0.5

### Etapa 7: Escalamiento
```
por_unidad = max(round(volumen / scale_n / 0.01) × 0.01, 0.01)
volumen_real = por_unidad × scale_n
volumen_real = min(volumen_real, profile.max_volume)
```

---

## 7. Market Memory

Base de datos SQLite (`data/market_memory.db`) que registra interacciones del precio con niveles de swing.

**Funcionalidad:**
- Registra cada toque de swing point con resultado: "bounce" o "break"
- Trackea visitas, bounces, breaks, toques totales por (símbolo, precio)
- Asocia tipos de patrón a cada nivel
- Alimentado en cada evaluación desde los swings del LTF

**Capacidades:**
- `get_nearby_levels()`: niveles dentro de 2× ATR del precio actual
- `get_level_bias()`: determina si hay soporte (BULLISH_BIAS), resistencia (BEARISH_BIAS) o ambos (RANGE_BIAS)
- Niveles consolidados: ≥ 3 toques y confiabilidad ≥ 0.5, fusionados dentro de 0.3 ATR
- Cache en memoria (1000 entradas), invalidación en escritura
- Limpieza automática de entradas > 30 días

**Uso en el pipeline:**
1. Se registran swings en cada evaluación
2. Antes de operar, `get_level_bias()` ajusta la convicción
3. Soporte cerca + BUY → ×1.2; Soporte cerca + SELL → ×0.5
4. Resistencia cerca + SELL → ×1.2; Resistencia cerca + BUY → ×0.5
5. Rango detectado → ×0.8

---

## 8. Meta-Learner

Base de datos SQLite (`data/meta_learning.db`) que registra cada trade y ajusta automáticamente los multiplicadores de patrón por régimen.

**Registro (TradeRecord):**
- Símbolo, dirección, entrada/salida, volumen, P&L
- Score, convicción, régimen, sesión
- Patrón primario, todos los patrones encontrados
- Confianza del régimen, razón de salida, duración

**Análisis (cada 4 horas, mínimo 10 trades):**
- Win rate y profit factor por régimen
- Mejor/peor patrón por régimen
- Win rate y profit promedio por patrón

**Ajustes automáticos:**
- Si win_rate del régimen < 30% (≥5 trades): reduce todos los multiplicadores ×0.85 (mín 0.3)
- Si win_rate del régimen > 70% (≥5 trades): aumenta todos los multiplicadores ×1.1 (máx 2.0)
- Sugerencias de ajuste por patrón (logueadas, no auto-aplicadas): si win_rate < 25% y profit negativo → sugiere reducir peso; si win_rate > 75% y profit positivo → sugiere aumentar peso

---

## 9. Pipeline Completo

```
1. Vela 5min → _on_new_candle → _evaluate()
2. risk_manager.can_trade() — límite diario de pérdidas
3. Por cada símbolo:
   a. DataFrames: H4..M1 (300 velas cada uno)
   b. NewsCalendar: evento de alto impacto activo?
   c. RegimeDetector: → RegimeContext
   d. SessionProfiler: → SessionProfile
   e. StrategyEngine.evaluate_adaptive():
      - MarketAnalyzer → MarketContext
      - PatternDetector → todos los patrones
      - _evaluate_direction(BUY) + _evaluate_direction(SELL)
      - DistributionalScorer → convicción
      - TradingSignal final
   f. MarketMemory: registrar swings, ajustar convicción
   g. ContinuousDecider → ContinuousDecision
   h. _manage_symbol_position(): BE, trailing, TP expansion, scale-out, reversals
   i. Si convicción ≥ 30% y debe_operar:
      - Verificar sesión, spread, SL mínimo
      - Calcular volumen (7 etapas)
      - Escalar en n órdenes pendientes
      - place_pending_entry() / execute_signal()
4. Cada 1min → _manage_positions_light(): BE + TP expansion
5. Cada 4h → meta_learner.analyze_performance()
```

---

## 10. Flujo de Velas

| Timeframe | Acción |
|-----------|--------|
| **1min** | `_manage_positions_light()` — BE/TP check rápido sin ATR |
| **5min** | `_evaluate()` — pipeline completo de detección, scoring, decisión |
| **3min, 10min, 15min, 30min, 1H, etc.** | Solo logueo de nueva vela (sin acción) |

---

## 11. Mejoras y Correcciones Recientes

### Bug Fix: price_diff en SL/TP (CRÍTICO)
**Problema**: En `order_executor.py`, los métodos `place_pending_entry()`, `place_limit_order()` y `execute_signal()` calculaban SL/TP desplazándolos por `price_diff = entry_price - market_price`. Cuando una orden SELL LIMIT se colocaba lejos del mercado (ej. entry 4564.81, bid 4577.48), el `price_diff` negativo (-12.67) movía el SL **debajo** del entry — el lado equivocado para una SELL.

**Solución**: Reemplazar el desplazamiento por precio con recálculo por distancia desde el nuevo precio:
```python
# Antes (bug):
price_diff = entry_price - market_price
adjusted_sl = signal.stop_loss + price_diff  # ← SL se desplaza incorrectamente

# Después (fix):
sl_dist = abs(signal.stop_loss - signal.entry_price)
adjusted_sl = entry_price - sl_dist if is_long else entry_price + sl_dist
```

Afectaba a:
- `place_pending_entry()` — órdenes pendientes LIMIT/STOP
- `place_limit_order()` — órdenes LIMIT por gap
- `execute_signal()` — órdenes de mercado (corregido también, aunque el fallback bare+modify ya estaba bien)

### Mejora: BE en cada vela de 1min
**Problema**: El BE solo se revisaba cada 5 minutos (en `_manage_symbol_position`). En mercados rápidos, el precio podía alcanzar 30% del TP y revertirse dentro de la misma vela de 5min, perdiendo la oportunidad de activar BE.

**Solución**: Nuevo método `_manage_positions_light()` que se ejecuta en cada vela de 1 minuto y revisa:
- BE activation (≥ 30% del TP)
- TP expansion (≥ 75% del TP)
- Limpieza de posiciones desaparecidas (SL/TP hit)

Sin ATR ni dataframes — solo lectura de posiciones MT5 y aritmética.

### Mejora: Market Memory como filtro de convicción
Se alimenta la base de datos de niveles en cada evaluación. Antes de decidir una entrada, se consulta `get_level_bias()` para ajustar la convicción según soportes/resistencias cercanas.

### Mejora: Session Filter Override
Los patrones evitados por sesión ahora se permiten si la convicción es ≥ 50% (antes se bloqueaban estrictamente).

### Mejora: TP Expansion para Runners
Cuando el precio alcanza 75% del TP original y BE está activo, el TP se expande a 1.5× la distancia original.

### Mejora: Meta-Learner
Sistema completo de registro de trades, análisis periódico (cada 4h) y ajuste automático de multiplicadores de patrón por régimen.

---

## 12. Archivos Relevantes

| Archivo | Propósito | Líneas |
|---------|-----------|--------|
| `src/bot.py` | Orquestador principal, pipeline completo | ~1365 |
| `src/core/strategy_engine.py` | TradingSignal, evaluate_adaptive(), scoring | ~1362 |
| `src/core/regime_detector.py` | 6 regímenes con detección por ADX/ATR/EMA | ~330 |
| `src/core/continuous_decision.py` | ContinuousDecision con convicción y volumen | ~195 |
| `src/core/session_profiler.py` | 7 sesiones con patrones preferidos/evitados | ~261 |
| `src/core/market_memory.py` | SQLite de niveles de precio con confiabilidad | ~280 |
| `src/core/meta_learner.py` | SQLite de trades + análisis + auto-ajuste | ~482 |
| `src/core/distributional_score.py` | Score distribuido con convergencia/convicción | ~195 |
| `src/core/adaptive_scoring.py` | Pesos adaptativos por régimen | ~195 |
| `src/core/pattern_detector.py` | 24 tipos de patrones SMC | ~400 |
| `src/executor/order_executor.py` | Ejecución de órdenes con SL/TP corregido | ~453 |
| `src/risk/fixed_risk_manager.py` | Gestión de riesgo con ajuste por convicción | ~200 |
| `config/strategy.json` | Configuración de estrategia, regímenes, sesiones | ~200 |

---

## 13. Configuración Rápida

```bash
python -m src.bot
```

Requiere:
- Python 3.14+
- MetaTrader 5 (versión 5.0.5735+)
- Cuenta Exness MT5 demo (o cualquier broker con MT5)
- Paquetes: `MetaTrader5`, `pandas`, `loguru`, `numpy`

Bases de datos creadas automáticamente en `data/`:
- `trading_state.db` — estado de posiciones y batches
- `market_memory.db` — niveles de precio históricos
- `meta_learning.db` — historial de trades y análisis
