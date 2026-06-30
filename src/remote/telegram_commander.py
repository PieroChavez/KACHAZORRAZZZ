"""Telegram remote control for the trading bot."""
import threading
import time
import os
import requests
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from src.main import TradingBot


API_URL = "https://api.telegram.org/bot{token}/{method}"
POLL_INTERVAL = 2
AUTHORIZED_CHATS: set[int] = set()


class TelegramCommander:
    def __init__(self, bot: "TradingBot", token: str, chat_id: Optional[int] = None):
        self.bot = bot
        self.token = token
        if chat_id:
            AUTHORIZED_CHATS.add(chat_id)
        self._offset = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _call(self, method: str, read_timeout: int = 15, **kwargs) -> Optional[dict]:
        try:
            r = requests.post(
                API_URL.format(token=self.token, method=method),
                json=kwargs,
                timeout=(10, read_timeout),
            )
            data = r.json()
            if data.get("ok"):
                return data
            logger.warning(f"[Telegram] API error: {data}")
        except requests.Timeout:
            pass
        except Exception as e:
            logger.warning(f"[Telegram] Request failed: {e}")
        return None

    def send_message(self, text: str, chat_id: Optional[int] = None):
        for cid in (AUTHORIZED_CHATS if chat_id is None else [chat_id]):
            self._call("sendMessage", chat_id=cid, text=text, parse_mode="HTML")

    def _process_command(self, text: str, chat_id: int):
        AUTHORIZED_CHATS.add(chat_id)
        parts = text.strip().lower().split()
        cmd = parts[0] if parts else ""

        if cmd == "/start":
            self.send_message(
                "🤖 <b>Trading Bot Control</b>\n\n"
                "Comandos disponibles:\n"
                "/status — estado de cuenta y bot\n"
                "/positions — posiciones abiertas\n"
                "/cancel_all — cancela órdenes pendientes\n"
                "/stop — detiene el bot\n"
                "/restart — reinicia el bot\n"
                "/set_lot 0.2 — cambia el lotaje\n"
                "/set_be 200 — cambia distancia breakeven\n"
                "/set_trail 350 — cambia distancia trailing\n"
                "/help — este mensaje",
                chat_id,
            )
        elif cmd == "/help":
            self.send_message(
                "/status — estado\n/positions — posiciones\n/cancel_all — cancela todo\n"
                "/stop — detener\n/restart — reiniciar\n"
                "/set_lot X — cambiar lotaje\n"
                "/set_be N — distancia breakeven\n"
                "/set_trail N — distancia trailing",
                chat_id,
            )
        elif cmd == "/status":
            self._cmd_status(chat_id)
        elif cmd == "/positions":
            self._cmd_positions(chat_id)
        elif cmd == "/cancel_all":
            self._cmd_cancel_all(chat_id)
        elif cmd == "/stop":
            self._cmd_stop(chat_id)
        elif cmd == "/restart":
            self._cmd_restart(chat_id)
        elif cmd == "/set_lot" and len(parts) >= 2:
            self._cmd_set_lot(parts[1], chat_id)
        elif cmd == "/set_be" and len(parts) >= 2:
            self._cmd_set_param("BE_DISTANCE_PIPS", parts[1], "breakeven", chat_id)
        elif cmd == "/set_trail" and len(parts) >= 2:
            self._cmd_set_param("TRAIL_DISTANCE_PIPS", parts[1], "trailing", chat_id)
        else:
            self.send_message(f"Comando no reconocido: {text}", chat_id)

    def _cmd_status(self, chat_id: int):
        try:
            info = self.bot.mt5.get_account_info()
            if not info:
                self.send_message("❌ No se pudo obtener info de cuenta", chat_id)
                return
            running = "✅ ACTIVO" if getattr(self.bot, "running", False) else "⏸️ DETENIDO"
            lines = [
                f"<b>📊 Estado del Bot</b>",
                f"Estado: {running}",
                f"Balance: ${info.get('balance', 0):.2f}",
                f"Equity: ${info.get('equity', 0):.2f}",
                f"Profit: ${info.get('profit', 0):+.2f}",
                f"Margen libre: ${info.get('margin_free', 0):.2f}",
            ]
            positions = self.bot.mt5.get_positions()
            lines.append(f"Posiciones abiertas: {len(positions)}")
            for sym, sd in getattr(self.bot, "symbols", {}).items():
                status = sd.get("engine", "").get_status() if hasattr(sd.get("engine", ""), "get_status") else {}
                if isinstance(status, dict):
                    lines.append(f"🔹 {sym}: {status.get('active_packs', 0)} packs, {status.get('active_fractals', 0)} fractales")
            self.send_message("\n".join(lines), chat_id)
        except Exception as e:
            self.send_message(f"❌ Error: {e}", chat_id)

    def _cmd_positions(self, chat_id: int):
        try:
            positions = self.bot.mt5.get_positions()
            if not positions:
                self.send_message("📭 Sin posiciones abiertas", chat_id)
                return
            lines = [f"<b>📈 Posiciones ({len(positions)})</b>"]
            for p in positions:
                lines.append(
                    f"#{p['ticket']} {p['symbol']} {'🟢BUY' if p['type']=='buy' else '🔴SELL'} "
                    f"vol={p['volume']} entry={p['price_open']} SL={p.get('sl',0)} "
                    f"profit={p.get('profit',0):+.2f}"
                )
            self.send_message("\n".join(lines), chat_id)
        except Exception as e:
            self.send_message(f"❌ Error: {e}", chat_id)

    def _cmd_cancel_all(self, chat_id: int):
        try:
            import MetaTrader5 as mt5
            orders = mt5.orders_get()
            if not orders:
                self.send_message("✅ Sin órdenes pendientes", chat_id)
                return
            cancelled = 0
            for o in orders:
                result = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    cancelled += 1
            self.send_message(f"✅ {cancelled} órdenes canceladas", chat_id)
        except Exception as e:
            self.send_message(f"❌ Error cancelando: {e}", chat_id)

    def _cmd_stop(self, chat_id: int):
        self.bot.running = False
        self.send_message("⏹️ Bot deteniéndose...", chat_id)

    def _cmd_restart(self, chat_id: int):
        self.send_message("🔄 Reiniciando bot...", chat_id)
        self.bot.running = False
        self._running = False
        import sys, subprocess
        subprocess.Popen([sys.executable, "-c", """
import time, sys, os
time.sleep(3)
os.execv(sys.executable, [sys.executable, '-m', 'src.main'] + sys.argv[1:])
"""])
        self.send_message("🔄 Bot reiniciado", chat_id)

    def _cmd_set_lot(self, val: str, chat_id: int):
        try:
            lot = float(val)
            for sym, sd in getattr(self.bot, "symbols", {}).items():
                eng = sd.get("engine")
                if eng and hasattr(eng, "_calc_volume"):
                    orig = eng._calc_volume
                    eng._calc_volume = lambda f, s=None, _orig=orig, _lot=lot: _lot
            self.send_message(f"✅ Lotaje cambiado a {lot}", chat_id)
        except ValueError:
            self.send_message(f"❌ Valor inválido: {val}", chat_id)

    def _cmd_set_param(self, attr: str, val: str, name: str, chat_id: int):
        try:
            n = int(val)
            guard = getattr(self.bot, "trailing_guard", None)
            if guard and hasattr(guard, attr):
                setattr(guard, attr, n)
                self.send_message(f"✅ Distancia {name} cambiada a {n} pips", chat_id)
            else:
                self.send_message(f"❌ No se pudo cambiar {name}", chat_id)
        except ValueError:
            self.send_message(f"❌ Valor inválido: {val}", chat_id)

    def _poll(self):
        while self._running:
            try:
                data = self._call(
                    "getUpdates",
                    read_timeout=30,
                    offset=self._offset,
                    timeout=30,
                    allowed_updates=["message"],
                )
                if data and "result" in data:
                    for update in data["result"]:
                        self._offset = update["update_id"] + 1
                        msg = update.get("message")
                        if msg and "text" in msg:
                            self._process_command(msg["text"], msg["chat"]["id"])
            except Exception as e:
                logger.warning(f"[Telegram] Poll error: {e}")
            time.sleep(POLL_INTERVAL)

    def _drain_pending_updates(self):
        """Drain old pending updates so they don't trigger commands on restart."""
        try:
            data = self._call("getUpdates", read_timeout=5, offset=0, timeout=5)
            if data and "result" in data:
                max_id = max((u["update_id"] for u in data["result"]), default=0)
                if max_id:
                    self._offset = max_id + 1
                    self._call("getUpdates", read_timeout=2, offset=self._offset, timeout=2)
                    logger.info(f"[Telegram] Drained {len(data['result'])} pending update(s)")
        except Exception:
            pass

    def start(self):
        if not self.token:
            logger.warning("[Telegram] No token configured, skipping")
            return
        self._drain_pending_updates()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True, name="TelegramCommander")
        self._thread.start()
        logger.info("[Telegram] Commander started")

    def stop(self):
        self._running = False
