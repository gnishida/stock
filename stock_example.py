"""
Generate training data for the stock price prediction
"""

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import yfinance as yf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MSFT", type=str, help="ticker symbol")
    args = parser.parse_args()

    # Get data on the specified ticker
    tickerData = yf.Ticker(args.symbol)

    # Get the historical prices for this ticker
    start = "2010-1-1"
    end = datetime.today().strftime("%Y-%m-%d")
    tickerDf = tickerData.history(period='1d', start=start, end=end)

    # 
    plt.plot(tickerDf["Close"])
    plt.show()


if __name__ == "__main__":
    main()
