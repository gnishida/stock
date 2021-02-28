import argparse
import copy
from datetime import datetime
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf


MODEL_FILE_NAME = "lstm_model.h5"


def predict(model_dir, symbol, past_days):
    # Load the model
    model = tf.keras.models.load_model("{}/{}".format(model_dir, MODEL_FILE_NAME))

    # Get data on the specified ticker
    tickerData = yf.Ticker(symbol)

    # Get the historical prices for this ticker
    start = "2010-1-1"
    end = datetime.today().strftime("%Y-%m-%d")
    tickerDf = tickerData.history(period='1d', start=start, end=end)

    predictedY = []
    pastMonthPrices = []
    for i in range(len(tickerDf)):
        value = tickerDf["Close"][i]

        if len(pastMonthPrices) == past_days:
            data = copy.deepcopy(pastMonthPrices)

            # Normalize training data
            denom = data[-1]
            normalized_data = [value / denom for value in data]

            x = numpy.array(normalized_data)
            X = numpy.zeros((1, 1, len(x)), dtype=float)
            X[0, 0, :] = x
            y = model.predict(X).flatten()
            predictedY.append(y[0])

            pastMonthPrices.pop(0)
        else:
            predictedY.append(value)
		
        pastMonthPrices.append(value)
	
    # Draw prediction
    plt.plot(tickerDf["Close"])
    plt.plot(predictedY)
    plt.show()


def main():	
    parser = argparse.ArgumentParser()
    parser.add_argument("--past-days", default="5", type=int, help="How many past days are used for prediction?")
    parser.add_argument("--symbol", default="MSFT", type=str, help="Ticker symbol")
    parser.add_argument('--model_dir', default="models", help="path to folder containing models")
    args = parser.parse_args()	

    predict(args.model_dir, args.symbol, args.past_days)
	

if __name__== "__main__":
    main()