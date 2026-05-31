"""
zigzag.py

Detector ZigZag para BOT2

Genera pivotes HIGH y LOW
similares al ZigZag de TradingView.

Autor: BOT2
"""

import pandas as pd

from core.logger import logger


class ZigZagDetector:

    def __init__(
        self,
        deviation=0.5,
        depth=10
    ):

        self.deviation = deviation
        self.depth = depth

        logger.info(
            f"ZigZag iniciado | "
            f"Deviation={deviation}% "
            f"Depth={depth}"
        )

    # =====================================================
    # PIVOTS
    # =====================================================

    def detect(
        self,
        df: pd.DataFrame
    ) -> list:

        pivots = []

        try:

            if len(df) < self.depth * 2:

                return pivots

            # ==========================================
            # BUSCAR FRACTALES
            # ==========================================

            for i in range(
                self.depth,
                len(df) - self.depth
            ):

                current_high = df.iloc[i]["high"]
                current_low = df.iloc[i]["low"]

                left_highs = df.iloc[
                    i - self.depth:i
                ]["high"]

                right_highs = df.iloc[
                    i + 1:i + 1 + self.depth
                ]["high"]

                left_lows = df.iloc[
                    i - self.depth:i
                ]["low"]

                right_lows = df.iloc[
                    i + 1:i + 1 + self.depth
                ]["low"]

                # ======================================
                # PIVOT HIGH
                # ======================================

                if (
                    current_high > left_highs.max()
                    and
                    current_high > right_highs.max()
                ):

                    pivots.append({

                        "index": i,

                        "time":
                            df.iloc[i]["time"],

                        "price":
                            current_high,

                        "type":
                            "HIGH"
                    })

                # ======================================
                # PIVOT LOW
                # ======================================

                if (
                    current_low < left_lows.min()
                    and
                    current_low < right_lows.min()
                ):

                    pivots.append({

                        "index": i,

                        "time":
                            df.iloc[i]["time"],

                        "price":
                            current_low,

                        "type":
                            "LOW"
                    })

            # ==========================================
            # ORDENAR
            # ==========================================

            pivots.sort(
                key=lambda x: x["index"]
            )

            # ==========================================
            # FILTRAR POR DEVIATION
            # ==========================================

            filtered = []

            if len(pivots) == 0:

                return filtered

            filtered.append(
                pivots[0]
            )

            for pivot in pivots[1:]:

                previous = filtered[-1]

                change_pct = abs(

                    (
                        pivot["price"]
                        -
                        previous["price"]
                    )

                    /

                    previous["price"]

                ) * 100

                if change_pct >= self.deviation:

                    if pivot["type"] != previous["type"]:

                        filtered.append(
                            pivot
                        )

            logger.success(
                f"ZigZag pivots: {len(filtered)}"
            )

            return filtered

        except Exception as e:

            logger.exception(
                f"Error ZigZag: {e}"
            )

            return pivots


# =====================================================
# HELPER
# =====================================================

def detect_zigzag(
    df,
    deviation=0.5,
    depth=10
):

    detector = ZigZagDetector(
        deviation=deviation,
        depth=depth
    )

    return detector.detect(df)