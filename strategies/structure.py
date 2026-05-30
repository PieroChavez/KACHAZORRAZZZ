"""
structure.py
Detección de estructura de mercado (SMC)
BOS + CHOCH + Swing Highs/Lows
"""

import pandas as pd

from core.logger import logger


class MarketStructure:

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self):

        logger.info(
            "MarketStructure inicializado"
        )

    # =====================================================
    # SWING HIGHS / LOWS
    # =====================================================

    @staticmethod
    def detect_swings(
        df: pd.DataFrame,
        lookback: int = 3
    ) -> pd.DataFrame:

        """
        Detecta swing highs y swing lows.

        Swing High:
            high mayor a velas vecinas

        Swing Low:
            low menor a velas vecinas
        """

        try:

            df = df.copy()

            df["swing_high"] = False
            df["swing_low"] = False

            # =============================================
            # LOOP
            # =============================================

            for i in range(
                lookback,
                len(df) - lookback
            ):

                current_high = df.iloc[i]["high"]
                current_low = df.iloc[i]["low"]

                # =========================================
                # VELAS IZQUIERDA / DERECHA
                # =========================================

                left_highs = df.iloc[
                    i - lookback:i
                ]["high"]

                right_highs = df.iloc[
                    i + 1:i + 1 + lookback
                ]["high"]

                left_lows = df.iloc[
                    i - lookback:i
                ]["low"]

                right_lows = df.iloc[
                    i + 1:i + 1 + lookback
                ]["low"]

                # =========================================
                # SWING HIGH
                # =========================================

                if (
                    current_high > left_highs.max()
                    and
                    current_high > right_highs.max()
                ):

                    df.loc[
                        df.index[i],
                        "swing_high"
                    ] = True

                # =========================================
                # SWING LOW
                # =========================================

                if (
                    current_low < left_lows.min()
                    and
                    current_low < right_lows.min()
                ):

                    df.loc[
                        df.index[i],
                        "swing_low"
                    ] = True

            logger.success(
                "Swings detectados correctamente"
            )

            return df

        except Exception as e:

            logger.exception(
                f"Error detect_swings(): {e}"
            )

            return df

    # =====================================================
    # BOS DETECTION
    # =====================================================

    @staticmethod
    def detect_bos(
        df: pd.DataFrame
    ) -> list:

        """
        Detecta Break Of Structure.

        BOS alcista:
            rompe swing high previo

        BOS bajista:
            rompe swing low previo
        """

        bos_list = []

        try:

            # =============================================
            # OBTENER SWINGS
            # =============================================

            swing_highs = df[
                df["swing_high"] == True
            ]

            swing_lows = df[
                df["swing_low"] == True
            ]

            # =============================================
            # LOOP
            # =============================================

            for i in range(20, len(df)):

                current_close = df.iloc[i]["close"]

                # =========================================
                # ÚLTIMO SWING HIGH
                # =========================================

                previous_highs = swing_highs[
                    swing_highs.index < df.index[i]
                ]

                if len(previous_highs) > 0:

                    last_swing_high = (
                        previous_highs.iloc[-1]["high"]
                    )

                    # =====================================
                    # BULLISH BOS
                    # =====================================

                    if current_close > last_swing_high:

                        bos_list.append({

                            "index": i,

                            "time":
                                str(df.index[i]),

                            "type":
                                "BULLISH_BOS",

                            "price":
                                current_close,

                            "broken_level":
                                last_swing_high
                        })

                # =========================================
                # ÚLTIMO SWING LOW
                # =========================================

                previous_lows = swing_lows[
                    swing_lows.index < df.index[i]
                ]

                if len(previous_lows) > 0:

                    last_swing_low = (
                        previous_lows.iloc[-1]["low"]
                    )

                    # =====================================
                    # BEARISH BOS
                    # =====================================

                    if current_close < last_swing_low:

                        bos_list.append({

                            "index": i,

                            "time":
                                str(df.index[i]),

                            "type":
                                "BEARISH_BOS",

                            "price":
                                current_close,

                            "broken_level":
                                last_swing_low
                        })

            logger.success(
                f"BOS detectados: {len(bos_list)}"
            )

            return bos_list

        except Exception as e:

            logger.exception(
                f"Error detect_bos(): {e}"
            )

            return bos_list

    # =====================================================
    # CHOCH DETECTION
    # =====================================================

    @staticmethod
    def detect_choch(
        bos_list: list
    ) -> list:

        """
        Detecta Change Of Character.

        Cambio de:
            bullish -> bearish
            bearish -> bullish
        """

        choch_list = []

        try:

            if len(bos_list) < 2:

                return choch_list

            # =============================================
            # LOOP
            # =============================================

            for i in range(1, len(bos_list)):

                previous = bos_list[i - 1]
                current = bos_list[i]

                # =========================================
                # CAMBIO DIRECCIÓN
                # =========================================

                if previous["type"] != current["type"]:

                    choch_list.append({

                        "time":
                            current["time"],

                        "index":
                            current["index"],

                        "type":
                            "CHOCH",

                        "from":
                            previous["type"],

                        "to":
                            current["type"],

                        "price":
                            current["price"]
                    })

            logger.success(
                f"CHOCH detectados: "
                f"{len(choch_list)}"
            )

            return choch_list

        except Exception as e:

            logger.exception(
                f"Error detect_choch(): {e}"
            )

            return choch_list

    # =====================================================
    # TREND DETECTION
    # =====================================================

    @staticmethod
    def detect_trend(
        bos_list: list
    ) -> str:

        """
        Detecta tendencia general.
        """

        try:

            if len(bos_list) == 0:

                return "RANGE"

            last_bos = bos_list[-1]

            if last_bos["type"] == "BULLISH_BOS":

                return "BULLISH"

            if last_bos["type"] == "BEARISH_BOS":

                return "BEARISH"

            return "RANGE"

        except Exception as e:

            logger.exception(
                f"Error detect_trend(): {e}"
            )

            return "RANGE"


# =========================================================
# EJEMPLO USO
# =========================================================

if __name__ == "__main__":

    from data.market_data import MarketData

    market = MarketData(
        symbol="XAUUSDm"
    )

    # =============================================
    # DOWNLOAD DATA
    # =============================================

    df = market.get_historical_data(
        timeframe="M15",
        bars=500
    )

    # =============================================
    # STRUCTURE
    # =============================================

    structure = MarketStructure()

    # =============================================
    # SWINGS
    # =============================================

    df = structure.detect_swings(df)

    # =============================================
    # BOS
    # =============================================

    bos = structure.detect_bos(df)

    # =============================================
    # CHOCH
    # =============================================

    choch = structure.detect_choch(bos)

    # =============================================
    # TREND
    # =============================================

    trend = structure.detect_trend(bos)

    # =============================================
    # RESULTS
    # =============================================

    print("\n========== TREND ==========")
    print(trend)

    print("\n========== BOS ==========")

    for item in bos[-5:]:

        print(item)

    print("\n========== CHOCH ==========")

    for item in choch[-5:]:

        print(item)



# =========================================================
# FUNCIONES DE ALTO NIVEL (para imports fáciles)
# =========================================================

def detect_structure(
    df: pd.DataFrame,
    lookback: int = 3
) -> dict:
    """
    Función principal para detectar estructura de mercado.
    
    Returns:
        dict con: df (con swings), bos, choch, trend
    """
    try:
        ms = MarketStructure()
        
        # 1. Detectar swings
        df_with_swings = ms.detect_swings(df.copy(), lookback)
        
        # 2. Detectar BOS
        bos_list = ms.detect_bos(df_with_swings)
        
        # 3. Detectar CHOCH
        choch_list = ms.detect_choch(bos_list)
        
        # 4. Detectar tendencia
        trend = ms.detect_trend(bos_list)
        
        logger.info("Estructura de mercado detectada completamente")
        
        return {
            "df": df_with_swings,
            "bos": bos_list,
            "choch": choch_list,
            "trend": trend
        }
        
    except Exception as e:
        logger.exception(f"Error en detect_structure(): {e}")
        return {
            "df": df,
            "bos": [],
            "choch": [],
            "trend": "ERROR"
        }