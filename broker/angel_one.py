import pyotp
from SmartApi import SmartConnect
from config.settings import AngelOneConfig, TradingConfig
from utils.logger import logger


class AngelOneBroker:
    """Angel One SmartAPI integration for order placement and portfolio management."""

    def __init__(self):
        self.api = SmartConnect(api_key=AngelOneConfig.API_KEY)
        self.session = None
        self.auth_token = None
        self.feed_token = None
        self.profile = None

    def login(self) -> bool:
        try:
            totp = pyotp.TOTP(AngelOneConfig.TOTP_SECRET).now()
            data = self.api.generateSession(
                AngelOneConfig.CLIENT_ID,
                AngelOneConfig.PASSWORD,
                totp,
            )

            if data["status"]:
                self.auth_token = data["data"]["jwtToken"]
                self.feed_token = self.api.getfeedToken()
                self.profile = self.api.getProfile(data["data"]["refreshToken"])
                logger.info(f"Logged in to Angel One as {AngelOneConfig.CLIENT_ID}")
                return True
            else:
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Angel One login error: {e}")
            return False

    def place_order(
        self,
        symbol: str,
        token: str,
        exchange: str,
        transaction_type: str,  # BUY or SELL
        quantity: int,
        price: float = 0,
        order_type: str = "MARKET",
        product_type: str = "INTRADAY",
        trigger_price: float = 0,
    ) -> dict | None:
        try:
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": transaction_type,
                "exchange": exchange,
                "ordertype": order_type,
                "producttype": product_type,
                "duration": "DAY",
                "quantity": str(quantity),
            }

            if order_type == "LIMIT":
                order_params["price"] = str(price)
            if order_type == "STOPLOSS_LIMIT":
                order_params["price"] = str(price)
                order_params["triggerprice"] = str(trigger_price)
            if order_type == "STOPLOSS_MARKET":
                order_params["triggerprice"] = str(trigger_price)

            result = self.api.placeOrder(order_params)
            logger.info(f"TRADE: {transaction_type} {quantity} x {symbol} @ {price or 'MARKET'} | Order ID: {result}")
            return {"order_id": result, "status": "placed", **order_params}

        except Exception as e:
            logger.error(f"Order placement failed for {symbol}: {e}")
            return None

    def place_bracket_order(
        self,
        symbol: str,
        token: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        stop_loss: float,
        target: float,
    ) -> dict | None:
        """Place a bracket order with built-in SL and target."""
        try:
            sl_points = abs(price - stop_loss)
            target_points = abs(target - price)

            order_params = {
                "variety": "ROBO",
                "tradingsymbol": symbol,
                "symboltoken": token,
                "transactiontype": transaction_type,
                "exchange": exchange,
                "ordertype": "LIMIT",
                "producttype": "BO",
                "duration": "DAY",
                "price": str(price),
                "squareoff": str(round(target_points, 2)),
                "stoploss": str(round(sl_points, 2)),
                "quantity": str(quantity),
            }

            result = self.api.placeOrder(order_params)
            logger.info(
                f"TRADE: BO {transaction_type} {quantity}x {symbol} @ {price} "
                f"SL: {stop_loss} TGT: {target} | Order ID: {result}"
            )
            return {"order_id": result, "status": "placed", "type": "bracket", **order_params}

        except Exception as e:
            logger.error(f"Bracket order failed for {symbol}: {e}")
            return None

    def place_gtt_order(
        self,
        symbol: str,
        token: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        price: float,
        trigger_price: float,
    ) -> dict | None:
        """Place GTT (Good Till Triggered) order for swing trades."""
        try:
            gtt_params = {
                "tradingsymbol": symbol,
                "symboltoken": token,
                "exchange": exchange,
                "transactiontype": transaction_type,
                "producttype": "DELIVERY",
                "price": str(price),
                "qty": str(quantity),
                "triggerprice": str(trigger_price),
                "disclosedqty": str(quantity),
            }

            result = self.api.gttCreateRule(gtt_params)
            logger.info(f"TRADE: GTT {transaction_type} {quantity}x {symbol} trigger@{trigger_price}")
            return {"rule_id": result, "status": "created", "type": "gtt"}

        except Exception as e:
            logger.error(f"GTT order failed for {symbol}: {e}")
            return None

    def get_positions(self) -> list[dict]:
        try:
            positions = self.api.position()
            if positions and positions.get("data"):
                return positions["data"]
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_holdings(self) -> list[dict]:
        try:
            holdings = self.api.holding()
            if holdings and holdings.get("data"):
                return holdings["data"]
            return []
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            return []

    def get_order_book(self) -> list[dict]:
        try:
            orders = self.api.orderBook()
            if orders and orders.get("data"):
                return orders["data"]
            return []
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        try:
            self.api.cancelOrder(order_id, variety)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_pnl(self) -> dict:
        positions = self.get_positions()
        total_pnl = 0.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0

        for pos in positions:
            try:
                pnl = float(pos.get("pnl", 0))
                total_pnl += pnl
                if int(pos.get("netqty", 0)) == 0:
                    realized_pnl += pnl
                else:
                    unrealized_pnl += pnl
            except (ValueError, TypeError):
                pass

        return {
            "total_pnl": round(total_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
        }

    def get_ltp(self, exchange: str, symbol: str, token: str) -> float | None:
        try:
            data = self.api.ltpData(exchange, symbol, token)
            if data and data.get("data"):
                return float(data["data"]["ltp"])
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol}: {e}")
        return None

    def search_symbol(self, symbol: str, exchange: str = "NSE") -> dict | None:
        """Search for symbol token (needed for placing orders)."""
        try:
            # Angel One uses instrument list - this is a simplified lookup
            # In production, download the full instrument list from:
            # https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json
            search_result = self.api.searchScrip(exchange, symbol)
            if search_result and search_result.get("data"):
                return search_result["data"][0]  # First match
        except Exception as e:
            logger.error(f"Symbol search failed for {symbol}: {e}")
        return None

    def logout(self):
        try:
            self.api.terminateSession(AngelOneConfig.CLIENT_ID)
            logger.info("Logged out from Angel One")
        except Exception as e:
            logger.error(f"Logout error: {e}")
