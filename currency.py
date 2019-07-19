# currency module provide convertion values and a map of the ETFs to currency.

BASE_CURRENCY = "USD"


def CurrencyForAsset(symbol: str) -> str:
    """Returns the currency for the asset given by symbol."""
    if symbol.endswith(".L"):
        return "GBP"
    return "USD"


def ConvertionRate(date: str, symbol: str, target_symbol: str = BASE_CURRENCY) -> float:
    """Returns the convertion rate from 'symbol' to 'target_symbol'."""
    if symbol == target_symbol:
        return 1.0

    # TODO: Download convertion rate online.
    return 1.0
