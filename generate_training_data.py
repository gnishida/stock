import argparse
import copy
from datetime import datetime
import pandas
import sklearn
import yfinance as yf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--past-days", default="5", type=int, help="How many past days are used for prediction?")
    parser.add_argument("--symbol", default="MSFT", type=str, help="Ticker symbol")
    parser.add_argument("--out", default="train_data.txt", type=str, help="Output file name")
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
        value = tickerDf["Close"][i]
        if len(pastMonthPrices) == args.past_days:
            data = copy.deepcopy(pastMonthPrices)

            # Normalize training data
            denom = data[-1]
            data.append(value)
            normalized_data = [value / denom for value in data]

            # Write data to the output file
            normalized_data_str = [str(v) for v in normalized_data]
            normalized_data_str = ",".join(normalized_data_str)
            file.write("{}\n".format(normalized_data_str))

            pastMonthPrices.pop(0)

        pastMonthPrices.append(value)
        
    file.close()


if __name__ == "__main__":
    main()
