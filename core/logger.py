from loguru import logger
import sys
import os

os.makedirs("logs", exist_ok=True)

logger.remove()

# Consola
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}"
)

# Archivo general
logger.add(
    "logs/bot_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="60 days",
    level="DEBUG",
    encoding="utf-8"
)

# Archivo solo errores
logger.add(
    "logs/errors_{time:YYYY-MM-DD}.log",
    level="ERROR",
    rotation="1 week"
)