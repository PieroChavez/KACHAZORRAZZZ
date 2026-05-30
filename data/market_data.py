"""
market_data.py
Módulo profesional de datos MT5
Optimizado para Trading Bot + IA
"""

import os
import pandas as pd
import MetaTrader5 as mt5

from loguru import logger


class MarketDataManager:

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self, symbol="XAUUSD"):

        self.symbol = symbol

        logger.info(
            f"MarketDataManager iniciado -> {symbol}"
        )

    # =========================================================
    # TIMEFRAME MAP
    # =========================================================

    @staticmethod
    def _resolve_timeframe(timeframe):

        tf_map = {

            "M1": mt5.TIMEFRAME_M1,
            "M2": mt5.TIMEFRAME_M2,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,

            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,

            "D1": mt5.TIMEFRAME_D1
        }

        return tf_map.get(timeframe.upper())

    # =========================================================
    # SYMBOL MAP
    # =========================================================

    @staticmethod
    def _resolve_symbol(symbol):

        alias = {

            "XAUUSD": [
                "XAUUSD",
                "XAUUSDm",
                "GOLD"
            ],

            "EURUSD": [
                "EURUSD",
                "EURUSDm"
            ],

            "BTCUSD": [
                "BTCUSD",
                "BTCUSDm"
            ]
        }

        return alias.get(symbol.upper(), [symbol])

    # =========================================================
    # VERIFY SYMBOL
    # =========================================================

    def verify_symbol(self):

        symbols = self._resolve_symbol(
            self.symbol
        )

        for symbol in symbols:

            info = mt5.symbol_info(symbol)

            if info is not None:

                if not info.visible:

                    mt5.symbol_select(symbol, True)

                logger.success(
                    f"Símbolo activo -> {symbol}"
                )

                return symbol

        logger.error(
            f"No existe símbolo válido: {self.symbol}"
        )

        return None

    # =========================================================
    # GET HISTORICAL DATA
    # =========================================================

    def get_historical_data(
        self,
        timeframe="H1",
        bars=500
    ):

        try:

            # =============================================
            # TIMEFRAME
            # =============================================

            tf = self._resolve_timeframe(
                timeframe
            )

            if tf is None:

                logger.error(
                    f"Timeframe inválido: {timeframe}"
                )

                return None

            # =============================================
            # SYMBOL
            # =============================================

            symbol = self.verify_symbol()

            if symbol is None:
                return None

            # =============================================
            # DOWNLOAD
            # =============================================

            rates = mt5.copy_rates_from_pos(
                symbol,
                tf,
                0,
                bars
            )

            if rates is None:

                logger.error(
                    f"MT5 ERROR: {mt5.last_error()}"
                )

                return None

            # =============================================
            # DATAFRAME
            # =============================================

            df = pd.DataFrame(rates)

            if df.empty:

                logger.error(
                    f"Sin datos {symbol} {timeframe}"
                )

                return None

            # =============================================
            # TIME
            # =============================================

            df["time"] = pd.to_datetime(
                df["time"],
                unit="s"
            )

            # =============================================
            # RENAME
            # =============================================

            df.rename(
                columns={
                    "tick_volume": "volume"
                },
                inplace=True
            )

            # =============================================
            # IMPORTANT COLUMNS
            # =============================================

            df = df[
                [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "spread"
                ]
            ]

            # =============================================
            # CLEAN
            # =============================================

            df.dropna(inplace=True)

            df = df[df["high"] >= df["low"]]
            df = df[df["open"] > 0]
            df = df[df["close"] > 0]

            # =============================================
            # RESET INDEX
            # =============================================

            df.reset_index(
                drop=True,
                inplace=True
            )

            logger.success(
                f"{symbol} {timeframe} -> "
                f"{len(df)} velas"
            )

            return df

        except Exception as e:

            logger.exception(
                f"Error get_historical_data(): {e}"
            )

            return None

    # =========================================================
    # MULTI TIMEFRAME
    # =========================================================

    def get_multi_timeframes(
        self,
        timeframes,
        bars=500
    ):

        result = {}

        for tf in timeframes:

            df = self.get_historical_data(
                timeframe=tf,
                bars=bars
            )

            if df is not None:

                result[tf] = df

        logger.success(
            f"Datasets descargados -> "
            f"{list(result.keys())}"
        )

        return result

    # =========================================================
    # SAVE DATASET
    # =========================================================

    def save_dataset(
        self,
        df,
        timeframe,
        folder="datasets"
    ):

        try:

            # =============================================
            # CREATE FOLDER
            # =============================================

            os.makedirs(
                folder,
                exist_ok=True
            )

            # =============================================
            # FILE NAME
            # =============================================

            filename = (
                f"{self.symbol}_{timeframe}.csv"
            )

            path = os.path.join(
                folder,
                filename
            )

            # =============================================
            # COPY DF
            # =============================================

            df_to_save = df.copy()

            # =============================================
            # FORCE DATETIME
            # =============================================

            df_to_save["time"] = pd.to_datetime(
                df_to_save["time"]
            )

            # =============================================
            # OVERWRITE WITH FRESH DATA
            # =============================================

            combined = df_to_save

            # =============================================
            # SORT AND RESET INDEX BEFORE SAVING
            # =============================================

            combined.sort_values(
                by="time",
                inplace=True
            )
            combined.reset_index(
                drop=True,
                inplace=True
            )

            # =============================================
            # SAVE CSV
            # =============================================

            combined.to_csv(
                path,
                index=False
            )

            logger.success(
                f"Dataset actualizado -> {path}"
            )

            return path

        except Exception as e:

            logger.exception(
                f"Error save_dataset(): {e}"
            )

            return None

    # =========================================================
    # FEATURES
    # =========================================================

    @staticmethod
    def create_basic_features(df):

        try:

            # =============================================
            # RETURNS
            # =============================================

            df["returns"] = (
                df["close"].pct_change()
            )

            # =============================================
            # BODY
            # =============================================

            df["body"] = (
                abs(df["close"] - df["open"])
            )

            # =============================================
            # RANGE
            # =============================================

            df["range"] = (
                df["high"] - df["low"]
            )

            # =============================================
            # BULLISH
            # =============================================

            df["bullish"] = (
                df["close"] > df["open"]
            ).astype(int)

            # =============================================
            # VOLATILITY
            # =============================================

            df["volatility"] = (
                df["high"] - df["low"]
            ).rolling(10).mean()

            logger.success(
                "Features creadas"
            )

            return df

        except Exception as e:

            logger.exception(
                f"Error create_basic_features(): {e}"
            )

            return df


# =============================================================
# EXPORT
# =============================================================

MarketData = MarketDataManager