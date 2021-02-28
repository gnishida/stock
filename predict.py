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
    start = "2019-1-1"
    end = datetime.today().strftime("%Y-%m-%d")
    tickerDf = tickerData.history(period="1d", start=start, end=end)

    trueY = []
    predictedY = []
    pastMonthPrices = []
    for i in range(len(tickerDf)):
        value = tickerDf["Close"][i]
        trueY.append(value)

        if len(pastMonthPrices) == past_days:
            data = copy.deepcopy(pastMonthPrices)

            # Normalize training data
            denom = data[-1]
            normalized_data = [value / denom for value in data]

            x = numpy.array(normalized_data)
            X = numpy.zeros((1, 1, len(x)), dtype=float)
            X[0, 0, :] = x
            predicted_y = model.predict(X).flatten()

            # Denormalize the prediction
            denormalized_predicted_y = predicted_y[0] * data[-1]
            predictedY.append(denormalized_predicted_y)

            pastMonthPrices.pop(0)
        else:
            predictedY.append(value)
		
        pastMonthPrices.append(value)
	
    # Draw prediction
    plt.plot(trueY, "b")
    plt.plot(predictedY, "r")
    plt.ylabel("{} stock prices".format(symbol))
    plt.title("Actual Stock vs Predictions comparison")
    plt.legend(["Actual", "Prediction"])
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
