from enum import Enum
from dataclasses import dataclass
from typing import Optional


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class OrderTypeDecision:
    order_type: OrderType
    entry_price: float
    reason: str
    confidence_adjustment: float = 1.0
    limit_distance_atr: Optional[float] = None
