import argparse
import copy
from datetime import datetime
import pandas
import yfinance as yf


PAST_RANGE = 28
FUTURE_RANGE = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MSFT", type=str, help="ticker symbol")
    parser.add_argument("--out", default="train_data.txt", type=str, help="output file")
    args = parser.parse_args()

    # Get data on the specified ticker
    tickerData = yf.Ticker(args.symbol)

    # Get the historical prices for this ticker
    start = "2010-1-1"
    end = datetime.today().strftime("%Y-%m-%d")
    tickerDf = tickerData.history(period='1d', start=start, end=end)

    # Generate training data
    file = open(args.out, "w")
    pastMonthPrices = []
    for i in range(len(tickerDf)):
        pastMonthPrices.append(tickerDf["Close"][i])
        
        if len(pastMonthPrices) == PAST_RANGE:
            if i + FUTURE_RANGE < len(tickerDf):
                # Normalize training data
                data = []
                for price in pastMonthPrices:
                    data.append("{:.4f}".format(price / tickerDf["Close"][i]))

                data = ",".join(data)
                ans = "{:.4f}".format(tickerDf["Close"][i + FUTURE_RANGE] / tickerDf["Close"][i])
                file.write("{},{}\n".format(data, ans))

            pastMonthPrices.pop(0)
        
    file.close()


if __name__ == "__main__":
    main()
