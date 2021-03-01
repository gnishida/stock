"""
Experiment for MACD strategy
"""

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import yfinance as yf


CONSISTENCY_DURATION = 3


def is_going_up(data, index):
    if index < CONSISTENCY_DURATION:
        return False
    
    for i in range(CONSISTENCY_DURATION):
        if data[index - CONSISTENCY_DURATION + i] >= data[index]:
            return False
        
    return True


def is_going_down(data, index):
    if index < CONSISTENCY_DURATION:
        return False
    
    for i in range(CONSISTENCY_DURATION):
        if data[index - CONSISTENCY_DURATION + i] <= data[index]:
            return False
    
    return True


def is_golden_cross(macd, macd_signal, index):
    if index < CONSISTENCY_DURATION:
        return False
    
    if not is_going_up(macd, index):
        return False
    
    for i in range(CONSISTENCY_DURATION - 1):
        if macd[index - i - 1] >= macd_signal[index - i - 1]:
            return False
    
    if macd[index] > macd_signal[index]:
        return True
    else:
        return False


def is_death_cross(macd, macd_signal, index):
    if index < CONSISTENCY_DURATION:
        return False
    
    if not is_going_down(macd, index):
        return False
    
    for i in range(CONSISTENCY_DURATION - 1):
        if macd[index - i - 1] <= macd_signal[index - i - 1]:
            return False
    
    if macd[index] < macd_signal[index]:
        return True
    else:
        return False


def run_simulation(data, macd, macd_signal):
    stock = 0
    cash = 0
    win_cnt = 0
    total_cnt = 0
    cost = 0
    buy_dates = {}
    sell_dates = {}
    for i in range(len(data)):
        if is_golden_cross(macd, macd_signal, i):
            buy_dates[data.keys()[i]] = data[i]
            stock += 1
            cash -= data[i]
            cost += data[i]
        if is_death_cross(macd, macd_signal, i) and stock > 0:
            sell_dates[data.keys()[i]] = data[i]
            if data[i] > cost / stock:
                win_cnt += 1
            total_cnt += 1
            cash += data[i] * stock
            stock = 0
            cost = 0
    
    return cash, float(win_cnt) / total_cnt, stock, buy_dates, sell_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MSFT", type=str, help="ticker symbol")
    args = parser.parse_args()

    # Get data on the specified ticker
    tickerData = yf.Ticker(args.symbol)

    # Get the historical prices for this ticker
    start = "2018-1-1"
    end = datetime.today().strftime("%Y-%m-%d")
    tickerDf = tickerData.history(period='1d', start=start, end=end)

    # Calculate MACD
    data = tickerDf["Close"]
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    # Simulate buy/sell
    profit, win_ratio, stock, buy_dates, sell_dates = run_simulation(data, macd, macd_signal)
    print("Profit: ${}".format(profit))
    print("Win ratio: {}%".format(win_ratio * 100))
    print("Remained #stock: {}".format(stock))

    fig, axs = plt.subplots(2)
    axs[0].plot(data, label="{} prices".format(args.symbol), color="k")
    axs[0].plot(buy_dates.keys(), buy_dates.values(), "b^")
    axs[0].plot(sell_dates.keys(), sell_dates.values(), "rv")
    axs[0].set_ylabel("Price ($)")
    axs[0].legend()
    axs[1].plot(macd, label="MACD", color="b")
    axs[1].plot(macd_signal, label="Signal Line", color="r")
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
