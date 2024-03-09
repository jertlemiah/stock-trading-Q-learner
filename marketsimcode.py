# An improved version of your marketsim code accepts a “trades” DataFrame (instead of a file). More info on the trades
# data frame is below. It is OK not to submit this file if you have subsumed its functionality into one of your other
# required code files. This file has a different name and a slightly different setup than your previous project.
# However, that solution can be used with several edits for the new requirements. 


# import datetime as dt
# import numpy as np
# import pandas as pd
# from util import get_data, plot_data
# from StrategyUtilities import *
from typing import List, NamedTuple

from matplotlib import pyplot as plt
# import numpy as np
import datetime as dt
import pandas as pd
from matplotlib import dates
from pandas.plotting import register_matplotlib_converters
# import random
# import indicators
from util import get_data

register_matplotlib_converters()

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


def portfolio_stats(port_values: pd.DataFrame):
    cumulative_return = (port_values[-1] / port_values[0]) - 1

    daily_returns = port_values[1:].values / port_values[:-1] - 1
    daily_returns = daily_returns[1:]
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    risk_adjusted_return = 0  # assume risk adjusted return is just 0
    sharpe_daily_adjustment = 252 ** 0.5
    sharpe_ratio = sharpe_daily_adjustment * (avg_daily_return - risk_adjusted_return) / std_daily_return
    return cumulative_return, avg_daily_return, std_daily_return, sharpe_ratio


def get_normalized_prices(symbols: list, start_date, end_date):
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]  # removes SPY, which is automatically added
    prices = prices / prices.iloc[0]
    return prices


def get_yesterday(today: dt.datetime, data: pd.DataFrame):
    loc = data.index.get_loc(today)
    yesterday = data.index[loc - 1] if loc > 0 else today
    return yesterday


def get_n_previous_days(today: dt.datetime, data: pd.DataFrame, n=1, include_today=True):
    loc = data.index.get_loc(today)
    offset = 1 if include_today else 0
    if loc - n + offset < 0:
        return None
    else:
        return data.index[loc - n + offset: loc + offset]


class Episode(NamedTuple):
    Data: pd.DataFrame
    Label: str
    Color: str
    Trades: pd.DataFrame
    # Trades: pd.DataFrame


def plot_episode_set(episodes: List[Episode], title, plot_vlines=True):
    fig, ax = plt.subplots()
    fig.autofmt_xdate(rotation=40)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(dates.MonthLocator())
    x_min = episodes[0].Data.iloc[0].name
    x_max = episodes[0].Data.iloc[-1].name
    ax.set_xlim([x_min, x_max])

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')

    if plot_vlines:
        colors = ['#7A19D9', '#FF0000', '#33CC33']
        plt.vlines(episodes[0].Trades.first_valid_index(), 1, 1, linestyles='dashdot', color='#0099FF', label='LONG Entry')
        plt.vlines(episodes[0].Trades.first_valid_index(), 1, 1, linestyles='dashed', color='#000000', label='SHORT Entry')
    vbar_height = 0.1

    for i in range(len(episodes)):
        plt.plot(episodes[i].Data.iloc[:], linewidth=1, label=episodes[i].Label, color=episodes[i].Color)

        short_dates = episodes[i].Trades.loc[episodes[i].Trades.iloc[:, 0] == 0].index
        height = (episodes[i].Data.iloc[:, 0].max() - episodes[i].Data.iloc[:, 0].min())
        mid = episodes[i].Data.loc[short_dates].iloc[:, 0]
        if plot_vlines:
            plt.vlines(short_dates, mid - height*vbar_height, mid + height*vbar_height, color='#0099FF', linestyles='dashdot')

        long_dates = episodes[i].Trades.loc[episodes[i].Trades.iloc[:, 0] == 2].index
        height = (episodes[i].Data.iloc[:, 0].max() - episodes[i].Data.iloc[:, 0].min())
        mid = episodes[i].Data.loc[long_dates].iloc[:, 0]
        if plot_vlines:
            plt.vlines(long_dates, mid + height*vbar_height, mid - height*vbar_height, color='#000000', linestyles='dashed')

    plt.grid(linestyle='--', color='#BBBBBB', linewidth=1)
    plt.legend()

    plt.savefig(f"images/{title}.png")


def compute_portvals(
        orders_df: pd.DataFrame,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_df: The dataframe containing the orders
    :type orders_df: pd.DataFrame
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # nonzero_orders = orders_df.loc[orders_df['Order'] != 0]
    # nonzero_orders = orders_df

    start_date = orders_df.first_valid_index()
    end_date = orders_df.last_valid_index()
    symbols = orders_df['Symbol'].unique()

    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices.loc[:, symbols]
    prices['Cash'] = 1  # this is so matrix multiplication works later

    # trades = pd.DataFrame(index=nonzero_orders.index, columns=['Symbol', 'Cash'])
    trades = pd.DataFrame.copy(prices)
    trades[:] = 0

    holdings = pd.DataFrame.copy(trades)
    holdings['Cash'][0] = start_val

    values = pd.DataFrame.copy(prices)

    for index, order in orders_df.iterrows():
        order: pd.DataFrame
        date = order.name
        symbol, shares = order

        # leverage = sum of abs value of investments / (sum of investments + cash)
        # if leverage > 3.0, discard trade

        impact_val = (1 + impact) if shares > 0 else (1 - impact)
        trades.loc[date, symbol] += shares
        trades.loc[date, 'Cash'] += -shares * prices.loc[date, symbol] * impact_val - commission

    i = 0
    for dateIndex, trade in trades.iterrows():
        if i == 0:
            holdings.iloc[i] += trade
        else:
            holdings.iloc[i] = holdings.iloc[i - 1] + trade
        i += 1

    values[:] = prices * holdings

    portvals = values.sum(axis=1)

    return portvals


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-10.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

        # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    cr, adr, sdr, sr = portfolio_stats(
        port_values=portvals
    )

    outputs = dict(
        num_days=141,
        last_day_portval=1026658.3265,
        sharpe_ratio=0.627643575702,
        avg_daily_ret=0.000222013722594,
    )

    num_days_bool = len(portvals) != outputs["num_days"]
    # last_day_portval_bool = abs(portvals[-1] - 1115569.2) > (0.001 * 1115569.2)
    last_day_portval_bool = (
            abs(portvals[-1] - outputs["last_day_portval"]) > 0.001
    )
    sharpe_ratio_bool = abs(sr - outputs["sharpe_ratio"]) > abs(
        0.001 * outputs["sharpe_ratio"]
    )
    avg_daily_ret_bool = abs(adr - outputs["avg_daily_ret"]) > abs(
        0.001 * outputs["avg_daily_ret"]
    )

    print(f"num_days_passed {not num_days_bool}, "
          f"\nlast_day_portval_passed {not last_day_portval_bool}, "
          f"\nsharpe_ratio_passed {not sharpe_ratio_bool}, "
          f"\navg_daily_ret_passed {not avg_daily_ret_bool}")

    # num_days = 245,
    # last_day_portval = 1115569.2,
    # sharpe_ratio = 0.612340613407,
    # avg_daily_ret = 0.00055037432146,

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
