class InstrumentContext:
    """
    Encapsulates instrument-level metadata and behavior.
    """

    def __init__(
        self,
        *,
        instrument_key: str,
        token: int,
        symbol: str,
        exchange: str,
        instrument_type: str,
        timezone: str,
        tick_size: float
    ):
        self.instrument_key = instrument_key
        self.token = token
        self.symbol = symbol
        self.exchange = exchange
        self.type = instrument_type.lower()
        self.timezone = timezone
        self.tick_size = tick_size

    # ---- Type checks ----

    def is_index(self) -> bool:
        return self.type == "index"

    def is_stock(self) -> bool:
        return self.type == "stock"

    def is_forex(self) -> bool:
        return self.type == "forex"

    # ---- Behavioral flags ----

    def supports_pre_open(self) -> bool:
        return self.exchange == "NSE" and self.is_index()

    def is_24x5_market(self) -> bool:
        return self.is_forex()

    def __repr__(self) -> str:
        return (
            f"InstrumentContext("
            f"token={self.token}, "
            f"symbol={self.symbol}, "
            f"type={self.type}, "
            f"exchange={self.exchange}"
            f")"
        )