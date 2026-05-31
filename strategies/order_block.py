"""
order_block.py

Order Block Detector V1

Basado en:

ZigZag
+
BOS

Genera zonas OB para dibujar
en Streamlit/Plotly.
"""

from loguru import logger


class OrderBlockDetector:

    def __init__(self):

        logger.info(
            "OrderBlockDetector iniciado"
        )

    # =====================================================
    # DETECT ORDER BLOCKS
    # =====================================================

    def detect(
        self,
        df,
        bos_list
    ):

        order_blocks = []

        try:

            if len(bos_list) == 0:

                return order_blocks

            # =============================================
            # RECORRER BOS
            # =============================================

            for bos in bos_list:

                bos_index = bos["index"]

                # =========================================
                # BULLISH BOS
                # =========================================

                if bos["type"] == "BULLISH_BOS":

                    for i in range(
                        bos_index - 1,
                        0,
                        -1
                    ):

                        candle = df.iloc[i]

                        # última vela bajista
                        if (
                            candle["close"]
                            <
                            candle["open"]
                        ):

                            order_blocks.append({

                                "type":
                                    "BULLISH_OB",

                                "index":
                                    i,

                                "time":
                                    candle["time"],

                                "high":
                                    float(
                                        candle["high"]
                                    ),

                                "low":
                                    float(
                                        candle["low"]
                                    ),

                                "mitigated":
                                    False
                            })

                            break

                # =========================================
                # BEARISH BOS
                # =========================================

                elif bos["type"] == "BEARISH_BOS":

                    for i in range(
                        bos_index - 1,
                        0,
                        -1
                    ):

                        candle = df.iloc[i]

                        # última vela alcista
                        if (
                            candle["close"]
                            >
                            candle["open"]
                        ):

                            order_blocks.append({

                                "type":
                                    "BEARISH_OB",

                                "index":
                                    i,

                                "time":
                                    candle["time"],

                                "high":
                                    float(
                                        candle["high"]
                                    ),

                                "low":
                                    float(
                                        candle["low"]
                                    ),

                                "mitigated":
                                    False
                            })

                            break

            logger.success(
                f"Order Blocks detectados: "
                f"{len(order_blocks)}"
            )

            return order_blocks

        except Exception as e:

            logger.exception(
                f"Error detectando OB: {e}"
            )

            return order_blocks


# =====================================================
# FUNCION SIMPLE
# =====================================================

def detect_order_blocks(
    df,
    bos_list
):

    detector = OrderBlockDetector()

    return detector.detect(
        df,
        bos_list
    )