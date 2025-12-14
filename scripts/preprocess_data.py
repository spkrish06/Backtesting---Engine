import yfinance as yf
import pandas as pd
import struct
import os
import argparse

def download_data(symbol, start_date, end_date):
    print(f"Downloading {symbol} from {start_date} to {end_date}...")
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        print("No data found!")
        return None
    return df

def convert_to_binary(df, symbol_id, output_file):
    print(f"Converting {len(df)} records to binary {output_file}...")
    
    with open(output_file, "wb") as f:
        # TickRecord Struct Layout Like:
        # timestamp (uint64), symbol_id (uint64), price (double), volume (double), flags (uint8), pad (7 bytes)
        # Format string: <QQddB7x
        
        for index, row in df.iterrows():
            # Convert timestamp to nanoseconds
            ts = int(index.value) # index is Timestamp
            
            # For simplicity, we use Close price as the tick price
            # Handle MultiIndex if present (yfinance sometimes returns it)
            if isinstance(row['Close'], pd.Series):
                 price = float(row['Close'].iloc[0])
                 volume = float(row['Volume'].iloc[0])
            else:
                 price = float(row['Close'])
                 volume = float(row['Volume'])
            
            flags = 1 # Trade
            
            # Pack
            packed = struct.pack("<QQddB7x", ts, symbol_id, price, volume, flags)
            f.write(packed)
            
    print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess market data")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-02-01", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="data/processed/market_data.bin", help="Output binary file")
    
    args = parser.parse_args()
    
    # It ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = download_data(args.symbol, args.start, args.end)
    if df is not None:
        convert_to_binary(df, 1, args.output)

if __name__ == "__main__":
    main()
