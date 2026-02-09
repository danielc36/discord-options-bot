import yfinance as yf

def get_stock_df(symbol, interval="1m"):
    ticker = yf.Ticker(symbol)

    # Map interval to proper period
    if interval == "1m":
        period = "1d"
    elif interval == "15m":
        period = "5d"
    else:
        period = "1d"

    df = ticker.history(period=period, interval=interval)
    df.dropna(inplace=True)
    return df


def get_option_chain(symbol):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return None, None, None

    expiry = expirations[0]
    chain = ticker.option_chain(expiry)

    return chain.calls, chain.puts, expiry
