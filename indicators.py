# Code implementing your indicators as functions that operate on DataFrames. There is no defined API for indicators.py,
# but when it runs, the main method should generate the charts that will illustrate your indicators in the report.
# from marketsimcode import *
import datetime
import datetime as dt
import numpy as np
import pandas as pd
# from util import get_data, plot_data
# from util import get_data, plot_data
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from util import get_data, plot_data
# import scipy.optimize as spo
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


def get_normalized_indicators(prices: pd.DataFrame):
    data = pd.DataFrame()

    data['Price'] = prices.iloc[:, 0]

    bbands = get_bollinger_bands(prices=data, span_days=20)
    cols = bbands.columns.difference(data.columns)
    data = pd.merge(data, bbands[cols], left_index=True, right_index=True, how='outer')

    ema = get_ema(prices=data, span_days=20)
    cols = ema.columns.difference(data.columns)
    data = pd.merge(data, ema[cols], left_index=True, right_index=True, how='outer')

    macd = get_macd(prices=data)
    cols = macd.columns.difference(data.columns)
    data = pd.merge(data, macd[cols], left_index=True, right_index=True, how='outer')

    cci = get_cci(prices=prices, span_days=20)
    cols = cci.columns.difference(data.columns)
    data = pd.merge(data, cci[cols], left_index=True, right_index=True, how='outer')

    return data


def get_bollinger_bands(prices: pd.DataFrame, span_days=20):
    data = pd.DataFrame()

    data['Price'] = prices.iloc[:, 0]
    data['SMA'] = data['Price'].rolling(span_days).mean()
    data['Upper Band'] = data['SMA'] + 2 * data['Price'].rolling(span_days).std()
    data['Lower Band'] = data['SMA'] - 2 * data['Price'].rolling(span_days).std()
    conditions = [
        data['Price'] > data['Upper Band'],
        data['Price'] < data['Lower Band'],
    ]
    choices = [-1, 1]
    data['BB-signal'] = np.select(conditions, choices, default=0)
    return data


def plot_bollinger_bands(prices: pd.DataFrame, span_days=20, fig_size=(15, 5)):
    fig, ax = plt.subplots(figsize=fig_size)

    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    legend = [f"{symbol} Price", f"{symbol} {span_days}-Day SMA", "Upper Band", "Lower Band"]
    title = f'{span_days}-Day Bollinger Bands for {symbol}'

    ax.title.set_text(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    x = prices.index
    data = get_bollinger_bands(prices, span_days=20)

    # plt.plot(prices)
    plt.plot(data['Price'], linestyle='-')
    plt.plot(data['SMA'], linestyle='--')
    plt.plot(data['Upper Band'], linestyle='--')
    plt.plot(data['Lower Band'], linestyle='--')
    plt.fill_between(x, data['Price'], data['Upper Band'],
                     where=data['Price'] > data['Upper Band'],
                     color='red',
                     alpha=0.4,
                     interpolate=True)
    plt.fill_between(x, data['Price'], data['Lower Band'],
                     where=data['Price'] < data['Lower Band'],
                     color='green',
                     alpha=0.4,
                     interpolate=True)
    plt.legend(legend)

    plt.savefig(f"{title}.png", bbox_inches='tight')
    return


def get_ema(prices: pd.DataFrame, span_days=20):
    data = pd.DataFrame()

    data['Price'] = prices.iloc[:, 0]
    data['EMA'] = data['Price'].ewm(span=span_days).mean()
    data['Price-to-EMA'] = data['Price'] / data['EMA']
    return data


def plot_exponential_moving_average(prices: pd.DataFrame, span_days=20, fig_size=(10, 5)):
    fig, axs = plt.subplots(figsize=fig_size, nrows=2)

    fig.autofmt_xdate(rotation=40)
    axs[0].xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    axs[0].xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    axs[0].set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    title1 = f'{span_days}-Day Exponential Moving Average for {symbol}'

    axs[0].title.set_text(title1)
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Normalized Price')
    axs[0].grid(linestyle='--', color='#BBBBBB', linewidth=1)

    data = get_ema(prices=prices,
                   span_days=span_days)
    x = prices.index
    y = data['Price']
    ema = data['EMA']

    # plt.plot(prices)
    axs[0].plot(y, linestyle='-', color='#7A19D9', linewidth=1)
    axs[0].plot(ema, linestyle='--', color='#FF0000', linewidth=1)
    axs[0].fill_between(x, y, ema,
                        where=y > ema,
                        color='red',
                        alpha=0.4,
                        interpolate=True)
    axs[0].fill_between(x, y, ema,
                        where=y < ema,
                        color='green',
                        alpha=0.4,
                        interpolate=True)
    axs[0].legend([f"{symbol} Price",
                   f"{symbol} {span_days}-Day EMA",
                   "Price > EMA",
                   "Price < EMA"])

    # plt.savefig(f"{title}.png")

    #  Plot 2: EMA to Price
    # fig, ax = plt.subplots(figsize=fig_size)

    fig.autofmt_xdate(rotation=40)
    axs[1].xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    axs[1].xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    axs[1].set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    title = f'{symbol} Price Normalized To {span_days}-Day EMA'

    axs[1].title.set_text(title)
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Normalized Price To EMA')
    axs[1].grid(linestyle='--', color='#BBBBBB', linewidth=1)

    # prices['EMA'] = prices.iloc[:, 0].ewm(span=span_days).mean()
    # # prices['EMA20-2'] = prices['EMA20']/prices.iloc[:, 0]
    # prices.iloc[:, 0] = prices.iloc[:, 0] / prices['EMA']
    # prices['EMA'] = prices['EMA'] / prices['EMA']
    x = prices.index
    y = data['Price-to-EMA']
    ema = data['EMA']/data['EMA']

    # plt.plot(prices)
    # plt.plot(y, linestyle='-')
    axs[1].plot(y, linestyle='-', color='#7A19D9', linewidth=1)
    axs[1].fill_between(x, y1=1, y2=y,
                        where=y > 1,
                        color='red',
                        alpha=0.4,
                        interpolate=True)
    axs[1].fill_between(x, y1=1, y2=y,
                        where=y < 1,
                        color='green',
                        alpha=0.4,
                        interpolate=True)
    axs[1].legend([f"{symbol} Price To EMA",
                   "Buy Signal Region",
                   "Sell Signal Region"])
    axs[1].plot(ema, linestyle='--', color='#FF0000', linewidth=1)

    # plt.savefig(f"{title}.png")
    plt.savefig(f"{span_days}-Day EMA for {symbol} With Price To EMA Subplot.png", bbox_inches='tight')
    return


def get_cci(prices: pd.DataFrame, span_days=20):
    data = pd.DataFrame()

    data['Price'] = prices.iloc[:, 0]
    data['SMA'] = data['Price'].rolling(span_days).mean()
    data['meanDev'] = data['Price'].rolling(span_days).apply(pd.Series.mad, raw=False)
    data['CCI'] = (data['Price'] - data['SMA']) / (0.015 * data['meanDev'])
    # conditions = [
    #     data['CCI'] < -100,
    #
    #     data['CCI'] > 100,
    # ]
    # choices = [1, -1]
    data['CCI-signal'] = (-1*data['CCI'].fillna(0)/100).astype(int)
    return data


def plot_commodity_channel_index(prices: pd.DataFrame, span_days, fig_size=(15, 5)):
    fig, ax = plt.subplots(figsize=fig_size)

    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    title = f'{span_days}-Day Commodity Channel Index for {symbol}'

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('CCI Value')

    data = get_cci(prices, span_days)

    x = prices.index
    y = data.iloc[:, 0]
    cci = data['CCI']

    plt.plot(data['CCI'], linestyle='-')
    plt.fill_between(x, 100, cci,
                     where=data['CCI-signal'] < 0,
                     color='red',
                     alpha=0.4,
                     interpolate=True)
    plt.fill_between(x, -100, cci,
                     where=data['CCI-signal'] > 0,
                     color='green',
                     alpha=0.4,
                     interpolate=True)

    plt.legend([f"CCI",
                "Sell Signal",
                "Buy Signal"])
    plt.axhline(y=100, linestyle='--', color='red', linewidth=1)
    plt.axhline(y=-100, linestyle='--', color='green', linewidth=1)

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    plt.savefig(f"{title}.png", bbox_inches='tight')
    return


def get_macd(prices: pd.DataFrame):
    data = pd.DataFrame()

    data['Price'] = prices.iloc[:, 0]
    data['EMA12'] = data['Price'].ewm(span=12).mean()
    data['EMA26'] = data['Price'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD-signal-line'] = data['MACD'].ewm(span=9).mean()
    conditions = [
        data['MACD'] > data['MACD-signal-line'],
        data['MACD'] < data['MACD-signal-line'],
    ]
    choices = [1, -1]
    data['MACD-signal'] = np.select(conditions, choices, default=0)
    return data


def plot_macd(prices: pd.DataFrame, fig_size=(15, 5)):
    fig, ax = plt.subplots(figsize=fig_size)

    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    title = f'Moving Average Convergence Divergence (MACD) for {symbol}'

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('MACD Value')

    data = get_macd(prices=prices)
    # prices['EMA12'] = prices.iloc[:, 0].ewm(span=12).mean()
    # prices['EMA26'] = prices.iloc[:, 0].ewm(span=26).mean()
    # prices['MACD'] = prices['EMA12'] - prices['EMA26']
    # prices['signal'] = prices['MACD'].ewm(span=9).mean()
    x = data.index
    y = data.iloc[:, 0]
    macd = data['MACD']
    signal_line = data['MACD-signal-line']
    signal = data['MACD-signal']


    # plt.plot(prices)

    plt.plot(macd, linestyle='-')
    plt.plot(signal_line, linestyle='--', color='orange')

    plt.fill_between(x, signal_line, macd,
                     where=signal < 0,
                     color='red',
                     alpha=0.4,
                     interpolate=True)
    plt.fill_between(x, signal_line, macd,
                     where=signal > 0,
                     color='green',
                     alpha=0.4,
                     interpolate=True)
    plt.legend([f"MACD",
                "Signal Line",
                "Sell Signal",
                "Buy Signal", ])

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    plt.savefig(f"{title}.png", bbox_inches='tight')
    return


def plot_rate_of_change(prices: pd.DataFrame, span_days, fig_size=(15, 5)):
    fig, ax = plt.subplots(figsize=fig_size)

    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = prices.iloc[0].name
    x_max = prices.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    symbol = prices.columns[0]
    title = f'{span_days}-Day Rate Of Change (ROC) Indicator for {symbol}'

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('ROC Value %')

    prices['shifted price'] = prices.iloc[:, 0].shift(span_days)
    prices['ROC'] = (prices.iloc[:, 0] / prices['shifted price']) * 100 - 100
    x = prices.index
    y = prices.iloc[:, 0]
    roc = prices['ROC']

    plt.plot(roc, linestyle='-')
    # plt.plot(signal, linestyle='--', color='orange')

    plt.fill_between(x, roc,
                     where=roc < 0,
                     color='red',
                     alpha=0.4,
                     interpolate=True)
    plt.fill_between(x, roc,
                     where=roc > 0,
                     color='green',
                     alpha=0.4,
                     interpolate=True)
    plt.legend([f"ROC",
                "Sell Signal",
                "Buy Signal", ])
    plt.axhline(y=0, linestyle='--', color='#FF0000', linewidth=1)

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    plt.savefig(f"{title}.png", bbox_inches='tight')
    return


def run():
    start_value = 100000
    symbol = "JPM"
    # symbol = "AAPL"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices = prices / prices.iloc[0]

    plot_bollinger_bands(prices, span_days=20)
    plot_exponential_moving_average(prices, span_days=20)
    plot_commodity_channel_index(prices, span_days=20)
    plot_rate_of_change(prices, span_days=20)
    plot_macd(prices)
    return

