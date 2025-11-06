import yfinance as yf
import pandas as pd
import numpy as np
from utils import download_historical_data

# Manually select pair by changing this index (0, 1, or 2)
selected_pair_idx = 0
pairs = [
    ['AAPL', 'ORCL'],
    ['ADBE', 'CSCO'],
    ['ADBE', 'MSFT'],
    ['CL', 'KMB'],
    ['INTC', 'ORCL'],
    ['KMB', 'MDLZ'],
    ['MMM', 'UPS']
]


def main():
    """Main pipeline execution."""
    print("=" * 50)
    print("    PAIRS TRADING PIPELINE")
    print("=" * 50)
    
    # Get selected pair info
    stock_a, stock_b = pairs[selected_pair_idx]
    print(f"\nSelected pair: {stock_a} & {stock_b}")
    
    # Download historical data
    selected_pair = pairs[selected_pair_idx]
    df = download_historical_data(selected_pair, years=15)
    
    print("\n" + "=" * 50)
    print("Pipeline initialization complete!")
    print("=" * 50)
    
    # Preview the data
    print(f"\nFirst 5 days:")
    print(df.head())
    print(f"\nLast 5 days:")
    print(df.tail())
    
    # TODO: Add further pipeline steps here
    # - Cointegration testing
    # - Spread calculation
    # - Signal generation
    # - Backtesting

if __name__ == "__main__":
    main()