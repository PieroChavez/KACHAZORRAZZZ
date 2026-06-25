# Informe Detallado del Proyecto: Trading Bot V3

**Fecha:** 24/06/2026
**Propósito:** Bot de trading automatizado para MetaTrader 5 basado en KACHAZORRAS

---

## 1. Descripción General

Trading Bot V3 es un sistema de trading algorítmico avanzado para **MetaTrader 5** que implementa una estrategia **Fractal Cascade** basada en conceptos institucionales (SMC/ICT). Opera principalmente sobre **XAUUSDc (Gold vs USD)** y está desplegado en una cuenta **REAL de Exness**.

El sistema realiza análisis multi-timeframe desde 4H hasta 1min, detectando patrones institucionales como Fair Value Gaps, Order Blocks, Breakers, Liquidity Sweeps, y Wyckoff, combinándolos con análisis de régimen de mercado, perfil de sesión, análisis de volumen (VSA) y correlación DXY.

---

## 2. Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.14.4 |
| Broker API | MetaTrader5 5.0.5735 |
| Datos | pandas, numpy |
| Persistencia | SQLite + aiosqlite (async) |
| Logging | loguru |
| Testing | pytest + pytest-asyncio |
| ML | NumPy puro (feedforward MLP) |
| Configuración | JSON + .env |

---

## 3. Estructura del Proyecto

```
trading-botV3/
├── .env                       # Credenciales MT5 (cuenta REAL activa)
├── requirements.txt           # Dependencias Python
├── config/                    # Configuración en JSON
│   ├── broker.json            # Path MT5 y símbolos
│   ├── strategy.json          # Config principal (~300 líneas)
│   ├── risk.json              # Parámetros de riesgo
│   └── news_events.json       # Calendario de noticias
├── src/                       # Código fuente principal
│   ├── main.py                # Punto de entrada (TradingBot)
│   ├── adapters/              # Conexión con MT5
│   ├── core/                  # Motor de análisis
│   ├── scoring/               # Sistema de puntuación adaptativa
│   ├── strategies/            # Estrategia Fractal Cascade
│   ├── learning/              # Meta-learning
│   ├── neural/                # Red neuronal (NumPy)
│   ├── scheduler/             # Scheduler de timeframes
│   └── utils/                 # Utilidades
├── scripts/                   # Scripts operativos (24)
├── tests/                     # Tests (7 archivos)
├── data/                      # Datos runtime (SQLite DBs)
├── models/                    # Modelos NN entrenados
└── logs/                      # Logs de sesiones
```

---

## 4. Arquitectura por Capas

### 4.1 Capa de Adaptación (`src/adapters/`)
- **MT5Client**: Wrapper sobre la API de MetaTrader 5. Maneja conexión/reconexión, resolución de símbolos (alias para USDOLLAR, US100, etc.), obtención de velas, información de cuenta, posiciones y órdenes.

### 4.2 Capa de Análisis Core (`src/core/`)

| Módulo | Líneas | Función |
|--------|--------|---------|
| `market_velocity.py` | ~100 | Detecta momentum, aceleración y clasifica ACCUMULATION/EXPANSION/NEUTRAL |
| `market_analyzer.py` | ~200 | Analiza estructura de mercado (swing points, BOS, zonas discount/premium) |
| `pattern_detector.py` | ~1065 | Detecta 15+ patrones SMC (FVG, OB, Breakers, Sweeps, Wyckoff, etc.) |
| `regime_detector.py` | ~332 | Clasifica 6 regímenes de mercado con confianza |
| `session_profiler.py` | ~276 | Perfil de sesiones (Asian, London, NY, Overlap, Close) |
| `strategy_engine.py` | ~1100 | Motor principal de evaluación adaptativa (~50 pesos de scoring) |
| `vsa.py` | ~193 | Volume Spread Analysis (confirmación, clímax, absorción) |
| `multi_timeframe.py` | ~50 | Fetcher multi-timeframe (4H a 2min) |

### 4.3 Capa de Scoring (`src/scoring/`)
- **AdaptiveScorer**: Transforma pesos estáticos en pesos dinámicos según régimen y sesión
- **DistributionalScorer**: Representa puntuación como distribución estadística (media, desviación, convergencia, convicción)
- **CandleClosureRatings**: Sistema de calificación de cierre de velas

### 4.4 Capa de Estrategia (`src/strategies/`)
- **FractalCascadeStrategy** (~526 lns): Estrategia principal. Escaneo macro de fractales (4H, 2H, 30min, 15min) + sub-fractales independientes en 5M. Detección BOS/CHoCH en niveles Fibonacci 0.72.
- **OrderPackManager** (~599 lns): Gestión de packs de órdenes individuales. Maneja entrada, SL/TP, breakeven (+10), trailing stops. Persistencia en SQLite para recuperación ante caídas.
- **FractalDB** (~177 lns): Base de datos SQLite para estado de fractales (activos, caché de swings, deduplicación).
- **FractalLearner** (~279 lns): Aprendizaje por refuerzo sobre resultados de fractales. Ajusta multiplicadores de volumen y lista negra de filtros.

### 4.5 Capa de Aprendizaje (`src/learning/`)
- **MetaLearner** (~627 lns): Sistema integral de meta-learning. Registra cada operación con contexto completo (régimen, patrones, convicción, resultado). Ajusta automáticamente pesos de scoring y multiplicadores de patrones según rendimiento histórico.

### 4.6 Capa Neural (`src/neural/`)
- **NeuralNetwork** (~185 lns): MLP feedforward en NumPy puro (sin PyTorch/TensorFlow). Capas ocultas ReLU, salida sigmoide. Backpropagation con mini-batch SGD.
- **Feature Engineering** (~126 lns): Codificación one-hot para régimen, sesión, patrón, dirección. Vector de features normalizado.
- **Trainer** (~119 lns): Pipeline de entrenamiento que carga registros, entrena la NN y guarda modelo + scaler en `models/<symbol>/`.
- **NeuralAdvisor** (~94 lns): Inferencia en vivo. Predice probabilidad de win y ajusta la convicción de la señal.

### 4.7 Capa de Scheduling (`src/scheduler/`)
- **TimeframeScheduler**: Thread en background que consulta MT5 cada ~5s, detecta nuevas velas y dispara callbacks por timeframe.

### 4.8 Utilidades (`src/utils/`)
- **helpers.py**: ATR, swing points, killzone detection, order flow, retracement classification
- **operation_registry.py** (~334 lns): Registro centralizado de operaciones. Persistencia SQLite por sesión. Genera resúmenes de sesión.
- **state_persistence.py** (~169 lns): Persistencia asíncrona SQLite para estado diario y recuperación ante caídas.

---

## 5. Flujo de Datos

```
MT5 Terminal
    │
    ▼
TimeframeScheduler (poll cada 5s)
    │
    ▼
MultiTimeframeFetcher (velas 4H → 1min)
    │
    ▼
MarketVelocityDetector (ACUMULACIÓN/EXPANSIÓN)
    │
    ▼
FractalCascadeStrategy.evaluate() (cada vela de 2min)
    │
    ├──► RegimeDetector (6 regímenes)
    ├──► SessionProfiler (sesión actual)
    ├──► PatternDetector (15+ patrones SMC)
    ├──► MarketAnalyzer (estructura, BOS, TRB)
    ├──► VSADetector (volumen)
    ├──► DXY Correlation
    │
    ▼
DistributionalScorer + AdaptiveScorer
    │
    ▼
NeuralAdvisor (ajuste de convicción)
    │
    ▼
OrderPackManager (ejecución, SL/TP, breakeven, trailing)
    │
    ▼
MetaLearner (análisis post-trade, ajuste de pesos)
```

---

## 6. Sistema de Scoring

El sistema utiliza **~50 pesos configurables** en `config/strategy.json` que incluyen:

- **Patrones**: FVG alcista/bajista, OB alcista/bajista, Breaker alcista/bajista, Sweep interno/externo, Wyckoff Spring/UTAD, Void Scalp, BOS Zone, Cycle, Sequence, Price Establishment, Sub-fractal, Interval Point, Pressure Zone, Harmonic Cycle
- **Contexto**: Alineación HTF, zona discount/premium, DXY alcista/bajista
- **VSA**: Confirmación de volumen, clímax, absorción, baja volatilidad
- **Sesión**: Multiplicadores de volatilidad por sesión
- **Régimen**: Multiplicadores de patrón por régimen (6×15+ combinaciones)

---

## 7. Regímenes de Mercado

El `RegimeDetector` clasifica 6 regímenes usando ATR ratio, ADX, pendientes de EMA y estructura de swings:

1. **STRONG_TREND_BULLISH** - Tendencia alcista fuerte
2. **STRONG_TREND_BEARISH** - Tendencia bajista fuerte
3. **RANGING** - Mercado lateral
4. **HIGH_VOLATILITY** - Alta volatilidad
5. **LOW_VOLATILITY** - Baja volatilidad
6. **TRANSITION** - Transición entre regímenes

---

## 8. Sesiones de Trading

El `SessionProfiler` mapea horas UTC a sesiones:

| Sesión | Horario UTC | Característica |
|--------|-------------|----------------|
| Asian | 00:00-07:00 | Baja volatilidad |
| London Open | 07:00-09:00 | Alta volatilidad |
| London Mid | 09:00-12:00 | Volatilidad media |
| NY Open | 12:00-15:00 | Máxima volatilidad |
| Overlap | 12:00-16:00 | London + NY |
| NY Afternoon | 15:00-21:00 | Volatilidad media-baja |
| Close | 21:00-24:00 | Cierre |

---

## 9. Configuración Actual

| Parámetro | Valor |
|-----------|-------|
| Modo | DEMO (en config, pero credenciales REAL en .env) |
| Símbolo activo | XAUUSDc |
| Símbolos inactivos | XAUUSDm, XAUEURm, NAS100, XAGUSDm |
| Riesgo por operación | 2% |
| Pérdida diaria máxima | 1% |
| Ratio RR mínimo | 2.0 |
| Broker | Exness (MT5) |
| Cuenta REAL | Login: 163101600 |

---

## 10. Testing

- **Framework**: pytest + pytest-asyncio
- **7 archivos de test**, 125+ tests totales
- Cobertura: adaptive_scoring, distributional_score, meta_learner, pattern_detector, regime_detector, session_profiler, strategy_engine
- Sin tests para: neural, fractal_cascade, order_pack, mt5_client (requieren conexión MT5)

---

## 11. Base de Datos (SQLite)

El sistema mantiene múltiples bases de datos SQLite por símbolo en `data/db/<symbol>/`:

| Base de Datos | Propósito |
|---------------|-----------|
| `trading_state.db` | Estado diario y recuperación |
| `order_packs.db` | Packs de órdenes activas |
| `fractal_state.db` | Estado de fractales |
| `fractal_learner.db` | Resultados de aprendizaje de fractales |
| `meta_learning.db` | Historial de trades para meta-learning |
| `market_memory.db` | Memoria de mercado global |

---

## 12. Scripts Operativos (24)

| Categoría | Scripts |
|-----------|---------|
| Análisis | `analyze_log.py`, `analyze_trades.py`, `query_db.py`, `seed_databases.py`, `verify_seed.py` |
| Trading | `copy_trader.py`, `cancel_all.py`, `check_positions.py`, `check_be.py` |
| Monitoreo | `check_account.py`, `check_status.py`, `mt5_status.py`, `history_check.py` |
| Reportes | `report.py`, `session_report.py`, `today_report.py`, `pnl_analysis.py`, `pnl_detailed.py` |
| Mantenimiento | `reset_bot.py`, `compare_strategies.py`, `debug_modify.py` |
| ML | `train_neural.py` |
| Visualización | `visualize_data.py` |

---

## 13. Red Neuronal

- **Arquitectura**: MLP feedforward (NumPy puro)
- **Activaciones**: ReLU (ocultas), Sigmoid (salida)
- **Features**: score_norm, conviction, regime_confidence, direction dummificada, regime dummificado, session dummificada, pattern_group dummificado
- **Target**: Probabilidad de win (0-1)
- **Optimización**: Mini-batch SGD con early stopping
- **Persistencia**: Pesos + scaler guardados en `models/<symbol>/trade_predictor.npz`
- **Uso en vivo**: `NeuralAdvisor` ajusta la convicción de la señal según predicción de la NN
- **Entrenamiento automático**: Se activa cuando hay ≥3 trades con profit

---

## 14. Observaciones Clave

1. **Metodología SMC Completa**: Implementación exhaustiva de conceptos ICT/SMC con más de 15 patrones institucionales.
2. **Auto-adaptativo**: El sistema ajusta sus propios pesos y parámetros mediante meta-learning y refuerzo.
3. **Persistencia Robusta**: Múltiples bases SQLite aseguran recuperación ante caídas y trazabilidad completa.
4. **Cuenta REAL en Producción**: El `.env` contiene credenciales activas de una cuenta REAL de Exness (modo configurado como DEMO).
5. **Sin dependencias externas de ML**: La red neuronal está implementada en NumPy puro, sin PyTorch/TensorFlow.
6. **Madurez del Proyecto**: Arquitectura bien modularizada, extensa configuración, sistema de logging completo, y múltiples scripts de análisis y monitoreo.
7. **Área de Mejora**: Sin CI/CD, sin Docker, cobertura de tests limitada para módulos core (fractal_cascade, order_pack, mt5_client).

---

## 15. Recomendaciones

1. **Separar credenciales**: Mover la cuenta REAL a un archivo separado y protegido, manteniendo DEMO por defecto.
2. **Aumentar cobertura de tests**: Especialmente para FractalCascadeStrategy, OrderPackManager y MT5Client.
3. **Containerización**: Agregar Docker para entornos reproducibles.
4. **Monitoreo**: Implementar alertas en tiempo real (Telegram/email) para eventos críticos.
5. **Versionado de modelos**: Mantener historial de modelos NN entrenados para poder revertir si es necesario.

---

*Informe generado el 24/06/2026 por análisis automatizado del código fuente.*
