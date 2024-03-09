# Code implementing a TheoreticallyOptimalStrategy (details below). It should implement testPolicy(), which returns a
# trades data frame (see below).
from marketsimcode import *
import datetime
import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
from util import get_data, plot_data
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from util import get_data, plot_data
import scipy.optimize as spo
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


def plot_episode_set(total_results: pd.DataFrame, legend, title):
    fig, ax = plt.subplots()
    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = total_results.iloc[0].name
    x_max = total_results.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')

    plt.plot(total_results.iloc[:, 0], '#7A19D9')
    plt.plot(total_results.iloc[:, 1], '#FF0000')
    plt.legend(legend)

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    # plt.show()
    plt.savefig(f"images/{title}.png")
    # plt.savefig("images/OptimizedFigureInNotes.png")


def plot_tos_closeup(data: pd.DataFrame, legend, title):
    fig, ax = plt.subplots()
    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d %Y"))
    ax.xaxis.set_major_locator(dates.DayLocator())
    x_min = data.iloc[0].name
    x_max = data.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')

    plt.plot(data.iloc[:, 0], '#7A19D9')
    plt.legend(legend)

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)

    # plt.show()
    plt.savefig(f"images/{title}.png")
    # plt.savefig("images/OptimizedFigureInNotes.png")


def testPolicy(symbol="JMP",
               sd=dt.datetime(2008, 1, 1),
               ed=dt.datetime(2009, 1, 1),
               sv=100000):
    """
    The optimal strategy is to buy when the price is going to increase and sell when the price is going to decrease
    Since we know the future, we can just compare the current day with the next day

    :return: df_trades, A single column data frame, indexed by date, whose values represent trades for each trading day
    (from the start date to the end date of a given period). Legal values are +1000.0 indicating a BUY of 1000 shares,
    -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING. Values of +2000 and -2000 for trades are also
    legal so long as net holdings are constrained to -1000, 0, and 1000. Note: The format of this data frame differs
    from the one developed in a prior project.
    :rtype: pd.DataFrame
    """
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices_norm = prices / prices.iloc[0]

    def conditions(x):
        if x['curr'] < x['prev'] and x['curr'] < x['next']:
            return 2000
        elif x['curr'] > x['prev'] and x['curr'] > x['next']:
            return -2000
        else:
            return 0

    df_trades = prices_norm.copy()
    df_trades['curr'] = prices_norm
    df_trades['prev'] = prices_norm.shift(1)
    df_trades['next'] = prices_norm.shift(-1)
    df_trades['Order'] = df_trades.apply(conditions, axis=1)

    df_trades = df_trades.loc[:, ['Order']]

    if (df_trades.idxmin() < df_trades.idxmax())[0]:
        df_trades.loc[df_trades.idxmin(), 'Order'] = df_trades.loc[df_trades.idxmin(), 'Order'] / 2
    else:
        df_trades.loc[df_trades.idxmax(), 'Order'] = df_trades.loc[df_trades.idxmax(), 'Order'] / 2

    return df_trades





def main():
    start_value = 100000
    symbol = "JPM"
    # symbol = "AAPL"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices_norm = prices / prices.iloc[0]
    episode = prices_norm.copy()

    df_trades = testPolicy(symbol=symbol,
                           sd=start_date,
                           ed=end_date,
                           sv=start_value)
    df_trades['Symbol'] = symbol
    df_trades = df_trades[['Symbol', 'Order']]
    optimal_values = compute_portvals(orders_df=df_trades,
                                      start_val=start_value,
                                      commission=0,
                                      impact=0)
    optimal_values = optimal_values / optimal_values.iloc[0]
    episode['TOS'] = optimal_values

    plot_episode_set(episode, [f"{symbol} Benchmark", f"{symbol} TOS"], f'Theoretical Optimal Strategy for {symbol}')
    plot_tos_closeup(episode[0:9], [f"{symbol} Price"], f'Close Up for Daily Prices of {symbol}')

    port_val_JPM, cumulative_return_JPM, avg_daily_return_JPM, std_daily_return_JPM, sharpe_ratio_JPM = portfolio_stats(episode[symbol])
    port_val, cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio = portfolio_stats(episode['TOS'])
    # avg_daily_ret, sharpe_ratio = get_stats(episode['TOS'])

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of JPM : {sharpe_ratio_JPM}")
    print()
    print(f"Cumulative Return of Fund: {cumulative_return}")
    print(f"Cumulative Return of JPM : {cumulative_return_JPM}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_return}")
    print(f"Standard Deviation of JPM : {std_daily_return_JPM}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_return}")
    print(f"Average Daily Return of JPM : {avg_daily_return_JPM}")
    print()
    print(f"Final Portfolio Value: {episode['TOS'][-1]}")

    return


def portfolio_characteristics(port_val: pd.DataFrame):
    # normed: pd.DataFrame = prices / prices.iloc[0]
    # alloced: pd.DataFrame = normed * allocs
    # #print("current allocation: \n" + str(alloced.iloc[0]))
    # # pos_vals = alloced * start_val
    # port_val: pd.DataFrame = alloced.sum(axis=1)

    daily_returns = port_val[1:].values / port_val[:-1] - 1
    daily_returns = daily_returns[1:]

    cumulative_return = (port_val[-1] / port_val[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    risk_adjusted_return = 0  # assume risk adjusted return is just 0
    sharpe_daily_adjustment = 252 ** 0.5
    sharpe_ratio = sharpe_daily_adjustment * (avg_daily_return - risk_adjusted_return) / std_daily_return
    return port_val, cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio
