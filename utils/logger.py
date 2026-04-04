import sys
from loguru import logger

# Remove default handler
logger.remove()

# Console handler with color
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
)

# File handler for all logs
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Separate file for trades only
logger.add(
    "logs/trades_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="90 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
    filter=lambda record: "TRADE" in record["message"],
)
