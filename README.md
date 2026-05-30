# Bot Trading - Documentación del Proyecto

## 📋 Descripción General

Sistema automatizado de trading que utiliza MetaTrader 5 (MT5) para ejecutar estrategias de trading. El proyecto está diseñado para analizar datos del mercado, ejecutar órdenes y realizar backtesting de estrategias.

---

## 📁 Estructura del Proyecto

```
bot_trading/
│
├── config/                 # Configuración del proyecto
│   ├── settings.py        # Configuraciones generales
│   └── credentials.py     # Credenciales MT5
│
├── core/                  # Funcionalidades principales
│   ├── __init__.py
│   ├── config.py          # Configuración del sistema
│   ├── logger.py          # Sistema de logging
│   ├── mt5_connector.py   # Conexión a MetaTrader 5
│   ├── market_data.py     # Obtención de datos del mercado
│   └── orders.py          # Gestión de órdenes
│
├── strategies/            # Estrategias de trading
│   └── strategy_base.py   # Clase base para estrategias
│
├── backtesting/           # Pruebas de estrategias
│   └── run_backtest.py    # Ejecutor de backtesting
│
├── logs/                  # Archivos de registro
│   └── trading.log        # Log del trading
│
├── data/                  # Datos del proyecto
│   └── historical/        # Datos históricos de precios
│
├── tests/                 # Pruebas unitarias
│   └── test_connection.py # Test de conexión MT5
│
├── main.py               # Archivo principal
├── requirements.txt      # Dependencias del proyecto
├── .env                  # Variables de entorno
├── .gitignore           # Archivos ignorados por Git
└── README.md            # Este archivo
```

---

## 🚀 Cómo Funciona el Proyecto

## streamlit run main.py = estoo es para iniciar el proyecto desde main



### 1. **Inicialización**

- `main.py` inicia la aplicación
- Se cargan las credenciales desde `config/credentials.py`
- Se establece conexión con MetaTrader 5

### 2. **Obtención de Datos**

- `core/market_data.py` obtiene datos históricos y en tiempo real
- Los datos se almacenan en `data/historical/`
- Se utilizan para análisis técnico

### 3. **Ejecución de Estrategias**

- Las estrategias heredan de `strategies/strategy_base.py`
- Se analizan los datos del mercado
- Se generan señales de compra/venta

### 4. **Gestión de Órdenes**

- `core/orders.py` ejecuta las órdenes en MT5
- Registra todas las transacciones
- Gestiona stop loss y take profit

### 5. **Backtesting**

- `backtesting/run_backtest.py` simula operaciones históricas
- Valida la rentabilidad de estrategias
- Genera reportes de desempeño

### 6. **Logging**

- `core/logger.py` registra eventos del sistema
- Los logs se guardan en `logs/trading.log`
- Facilita debugging y auditoría

---

## 📝 Notas y Anotaciones

### Configuración Inicial

- [ ] Instalar MetaTrader 5
- [ ] Obtener credenciales de cuenta de prueba
- [ ] Configurar archivo `.env` con credenciales
- [ ] Instalar dependencias: `pip install -r requirements.txt`

### Desarrollo

- [ ] Implementar `mt5_connector.py`
- [ ] Crear estrategia base en `strategy_base.py`
- [ ] Desarrollar estrategias específicas
- [ ] Escribir tests en `tests/`

### Pruebas

- [ ] Ejecutar backtesting con datos históricos
- [ ] Validar estrategias en cuenta de demostración
- [ ] Monitorear logs para errores

---

## 🔧 Dependencias

```
MetaTrader5
pandas
numpy
ta-lib (análisis técnico)
python-dotenv
```

Ver `requirements.txt` para la lista completa.

---

## ⚙️ Configuración

### Variables de Entorno (.env)

```
MT5_LOGIN=tu_login
MT5_PASSWORD=tu_contraseña
MT5_SERVER=tu_servidor
```

### Settings (config/settings.py)

```python
# Símbolos a operar
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY']

# Timeframes
TIMEFRAME = '1H'

# Tamaño de lote
LOT_SIZE = 0.1
```

---

## 📊 Módulos Principales

### `core/mt5_connector.py`

- Conecta con MetaTrader 5
- Obtiene datos de mercado
- Maneja la desconexión

### `core/market_data.py`

- Descarga datos históricos
- Calcula indicadores técnicos
- Prepara datos para análisis

### `core/orders.py`

- Crea órdenes de compra/venta
- Gestiona cierre de posiciones
- Registra histórico de operaciones

### `strategies/strategy_base.py`

- Clase base para todas las estrategias
- Define métodos: `analyze()`, `execute()`, `close()`

---

## 📚 Recursos Útiles

- [MetaTrader 5 Python Documentation](https://www.mql5.com/en/docs/integration/python_metatrader5)
- [Ta-Lib Documentation](https://ta-lib.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## ✅ Checklist de Desarrollo

### Phase 1: Configuración Básica

- [ ] Configurar estructura de directorios
- [ ] Instalar dependencias
- [ ] Conectar con MT5

### Phase 2: Desarrollo Core

- [ ] Implementar `mt5_connector.py`
- [ ] Implementar `market_data.py`
- [ ] Implementar `orders.py`

### Phase 3: Estrategias

- [ ] Crear estrategia base
- [ ] Desarrollar estrategias personalizadas
- [ ] Validar con backtesting

### Phase 4: Producción

- [ ] Tests completos
- [ ] Monitoreo en vivo
- [ ] Optimización de rendimiento

---

## 🐛 Troubleshooting

| Problema              | Solución                            |
| --------------------- | ----------------------------------- |
| No conecta MT5        | Verifica credenciales en `.env`     |
| Error de datos        | Revisa logs en `logs/trading.log`   |
| Estrategia lenta      | Optimiza cálculo de indicadores     |
| Órdenes no ejecutadas | Verifica disponibilidad de símbolos |

---

## 📞 Notas Personales

_(Espacio para tus anotaciones)_

---

**Última actualización:** 29 de mayo de 2026
**Estado del proyecto:** En desarrollo
