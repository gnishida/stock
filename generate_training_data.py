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
    pastPrices = []
    for i in range(len(tickerDf)):
        value = tickerDf["Close"][i]
        if len(pastPrices) == args.past_days + 1:
            data = []
            for i in range(len(pastPrices) - 1):
                roc = (pastPrices[i + 1] - pastPrices[i]) / pastPrices[i]
                data.append(roc)

            # Normalize training data
            max_roc = max(data)
            min_roc = min(data)
            data.append((value - pastPrices[-1]) / pastPrices[-1])
            normalized_data = [(value - min_roc) / (max_roc - min_roc) for value in data]

            # Write data to the output file
            normalized_data_str = [str(v) for v in normalized_data]
            normalized_data_str = ",".join(normalized_data_str)
            file.write("{}\n".format(normalized_data_str))

            pastPrices.pop(0)

        pastPrices.append(value)
        
    file.close()


if __name__ == "__main__":
    main()
