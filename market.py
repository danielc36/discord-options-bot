import yfinance as yf

def get_stock_df(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="5m")
    df.dropna(inplace=True)
    return df

def get_option_chain(symbol):
    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    if not expirations:
        return None, None, None

    expiry = expirations[0]  # nearest expiration
    chain = ticker.option_chain(expiry)

    return chain.calls, chain.puts, expiry
