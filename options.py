def filter_options(options_df, direction, current_price):
    if options_df is None or options_df.empty:
        return None

    # Basic cleanup
    options_df = options_df.dropna(subset=["volume", "openInterest"])
    options_df = options_df[
        (options_df["volume"] > 50) &
        (options_df["openInterest"] > 200)
    ]

    if options_df.empty:
        return None

    # Strike filtering
    if direction == "CALL":
        options_df = options_df[
            (options_df["strike"] >= current_price) &
            (options_df["strike"] <= current_price * 1.03)
        ]
    else:  # PUT
        options_df = options_df[
            (options_df["strike"] <= current_price) &
            (options_df["strike"] >= current_price * 0.97)
        ]

    if options_df.empty:
        return None

    # Best contract = most liquidity
    best = options_df.sort_values(
        by=["volume", "openInterest"],
        ascending=False
    ).iloc[0]

    return best
