import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import dateparser



def download_historical_data(selected_pair: list, years: int = 15) -> pd.DataFrame:
    """
    Download historical price data for the selected pair.
    
    Args:
        selected_pair: List containing two stock symbols [stock_a, stock_b]
        years: Number of years of historical data to download
    
    Returns:
        DataFrame with stock_a and stock_b close prices
    """

    stock_a, stock_b = selected_pair[0], selected_pair[1]
    
    print(f"\n=== Downloading data for {stock_a} & {stock_b} ===")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        # Download data for both stocks
        print(f"Fetching {stock_a} data...")
        data_a = yf.download(stock_a, start=start_date, end=end_date, progress=False)
        
        print(f"Fetching {stock_b} data...")
        data_b = yf.download(stock_b, start=start_date, end=end_date, progress=False)
        
        if data_a.empty or data_b.empty:
            raise ValueError("Could not retrieve data for one or both stocks.")
        
        # Extract Close prices and create DataFrame
        # Use .squeeze() to handle both single and multi-column DataFrames
        close_a = data_a['Close'].squeeze()
        close_b = data_b['Close'].squeeze()
        
        # Combine into single DataFrame with ticker names as columns
        df = pd.DataFrame({
    'stock_a': close_a,
    'stock_b': close_b
    })
        
        # Parse and format dates
        df.index = pd.to_datetime(df.index.map(lambda x: dateparser.parse(str(x))))
        
        # Sort from oldest to newest
        df = df.sort_index(ascending=True)
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"\n✓ Successfully downloaded:")
        print(f"  Combined dataset: {len(df)} trading days")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise