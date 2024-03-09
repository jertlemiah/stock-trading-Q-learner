#
#     Code implementing a ManualStrategy object (your Manual Strategy) in the strategy_evaluation/ directory.
#     It should implement testPolicy() which returns a trades data frame (see below). The main part of this code should
#     call marketsimcode as necessary to generate the plots used in the report. NOTE: You will have to create this file
#     yourself.
from datetime import datetime

import numpy as np
# import datetime as dt
import pandas as pd
# from pandas.plotting import register_matplotlib_converters
import random
import indicators
from marketsimcode import *
# import TheoreticallyOptimalStrategy as TOS
# from StrategyUtilities import *

register_matplotlib_converters()


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


class ManualStrategy(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.current_shares = 0
        self.trade_types = pd.DataFrame()

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jplauch3"

    def testPolicy(self,
                   symbol="JPM",
                   start_date=dt.datetime(2008, 1, 1),
                   end_date=dt.datetime(2009, 1, 1),
                   sv=100000,
                   use_CCI=True,
                   use_BB=True,
                   use_MACD=True):
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

        prices = get_normalized_prices(symbols=[symbol],
                                       start_date=start_date,
                                       end_date=end_date)
        ind = indicators.get_normalized_indicators(prices=prices)
        self.trade_types = pd.DataFrame(index=prices.index, columns=['OrderType', 'Shares'])

        # orders = pd.DataFrame(index=prices.index, columns=['Order', 'Date', 'Symbol', 'Shares'])
        orders = pd.DataFrame(index=prices.index, columns=['Order'])

        self.current_shares = 0

        for index, today in enumerate(prices.index):
            if self.enter_short(today=today, data=ind, use_CCI=use_CCI, use_BB=use_BB, use_MACD=use_MACD):
                action_type = 0  # Short
                if self.current_shares == 1000:
                    trade_shares = -2000
                else:
                    trade_shares = -1000
                self.current_shares += trade_shares
                # orders.loc[today] = ['SELL', today, symbol, trade_shares]
                orders.loc[today] = [trade_shares]
            elif self.enter_long(today=today, data=ind, use_CCI=use_CCI, use_BB=use_BB, use_MACD=use_MACD):
                action_type = 2  # Long
                if self.current_shares == -1000:
                    trade_shares = 2000
                else:
                    trade_shares = 1000
                self.current_shares += trade_shares
                # orders.loc[today] = ['BUY', today, symbol, trade_shares]
                orders.loc[today] = [trade_shares]
            elif self.exit_short(today=today, data=ind, use_CCI=use_CCI, use_BB=use_BB, use_MACD=use_MACD) or \
                self.exit_long(today=today, data=ind, use_CCI=use_CCI, use_BB=use_BB, use_MACD=use_MACD):
                action_type = 1  # Cash out
                trade_shares = -self.current_shares
                orders.loc[today] = [trade_shares]
                self.current_shares = 0
            else:
                action_type = 3
                trade_shares = 0
                orders.loc[today] = [0]
            self.trade_types.loc[today] = (action_type, trade_shares)

        return orders

    def is_already_shorting(self):
        return self.current_shares == -1000

    def is_already_longing(self):
        return self.current_shares == 1000

    def enter_short(self, today: dt.datetime, data: pd.DataFrame, use_CCI=True, use_BB=True, use_MACD=True):
        # When the stock is overbought & we think the price will drop
        # & we aren't already shorting
        if self.is_already_shorting():
            return False

        prev_n_days = get_n_previous_days(today, data, n=5, include_today=True)
        if prev_n_days is None:
            return False

        vote = 0
        if use_CCI:
            vote += ManualStrategy.get_cci_vote(data, prev_n_days, True)
        if use_BB:
            vote += ManualStrategy.get_bb_vote(data, prev_n_days, True)
        if use_MACD:
            vote += ManualStrategy.get_macd_vote(data, prev_n_days, True)

        return vote > 1

    def exit_short(self, today: dt.datetime, data: pd.DataFrame, use_CCI=True, use_BB=True, use_MACD=True):
        # When price is trending up and we are currently in a short position
        if not self.is_already_shorting():
            return False

        prev_n_days = get_n_previous_days(today, data, n=5, include_today=True)
        if prev_n_days is None:
            return False

        vote = 0
        if use_CCI:
            vote += ManualStrategy.get_cci_vote(data, prev_n_days, False)
        if use_BB:
            vote += ManualStrategy.get_bb_vote(data, prev_n_days, False)
        if use_MACD:
            vote += ManualStrategy.get_macd_vote(data, prev_n_days, False)

        prev_month = get_n_previous_days(today, data, n=120, include_today=True)
        if prev_month is not None:
            minima = data.loc[prev_month, 'Price'].idxmin()
            if minima == prev_n_days[-3]:
                vote += 2

        return vote > 0

    def enter_long(self, today: dt.datetime, data: pd.DataFrame, use_CCI=True, use_BB=True, use_MACD=True):
        if self.is_already_longing():
            return False

        prev_n_days = get_n_previous_days(today, data, n=5, include_today=True)
        if prev_n_days is None:
            return False

        vote = 0
        if use_CCI:
            vote += ManualStrategy.get_cci_vote(data, prev_n_days, False)
        if use_BB:
            vote += ManualStrategy.get_bb_vote(data, prev_n_days, False)
        if use_MACD:
            vote += ManualStrategy.get_macd_vote(data, prev_n_days, False)

        return vote > 1

    def exit_long(self, today: dt.datetime, data: pd.DataFrame, use_CCI=True, use_BB=True, use_MACD=True):
        if not self.is_already_longing():
            return False

        prev_n_days = get_n_previous_days(today, data, n=5, include_today=True)
        if prev_n_days is None:
            return False

        vote = 0
        if use_CCI:
            vote += ManualStrategy.get_cci_vote(data, prev_n_days, True)
        if use_BB:
            vote += ManualStrategy.get_bb_vote(data, prev_n_days, True)
        if use_MACD:
            vote += ManualStrategy.get_macd_vote(data, prev_n_days, True)

        prev_month = get_n_previous_days(today, data, n=120, include_today=True)
        if prev_month is not None:
            maxima = data.loc[:today, 'Price'].idxmax()
            if maxima == prev_n_days[-3]:
                vote += 2

        return vote > 0

    @staticmethod
    def get_cci_vote(data: pd.DataFrame, prev_n_days: int, is_selling: bool):
        prev_n_signals = data.loc[prev_n_days, 'CCI-signal'].to_numpy()
        is_selling = 1 if is_selling else -1

        if prev_n_signals[-1] * is_selling <= -2 and prev_n_signals[-1] * is_selling > prev_n_signals[-2] * is_selling:
            return 3
        elif np.array_equal(prev_n_signals[-5:], np.array([-1, -1, -1, 0, 0]) * is_selling):
            return 2
        elif np.array_equal(prev_n_signals[-5:], np.array([-1, -1, 0, 0, 0]) * is_selling):
            return 2
        elif np.array_equal(prev_n_signals[-4:], np.array([-1, -1, 0, 0]) * is_selling):
            return 0
        elif np.array_equal(prev_n_signals[-2:], np.array([-1, 0]) * is_selling):
            return 0
        elif prev_n_signals[-1] * is_selling > 0:
            return -1
        else:
            return 0

    @staticmethod
    def get_bb_vote(data: pd.DataFrame, prev_n_days: int, is_selling: bool):
        prev_n_signals = data.loc[prev_n_days, 'BB-signal'].to_numpy()
        is_selling = -1 if is_selling else 1
        if np.array_equal(prev_n_signals[-2:], np.array([1, 1]) * is_selling):
            return -2
        elif np.array_equal(prev_n_signals[-5:], np.array([-1, -1, -1, 0, 0]) * is_selling):
            return 3
        elif np.array_equal(prev_n_signals[-5:], np.array([-1, -1, 0, 0, 0]) * is_selling):
            return 3
        elif np.array_equal(prev_n_signals[-4:], np.array([-1, -1, 0, 0]) * is_selling):
            return 2
        # elif np.array_equal(prev_n_signals[-3:], np.array([-1, 0, 0]) * is_selling):
        #     return 0
        # elif np.array_equal(prev_n_signals[-2:], np.array([-1, 0]) * is_selling):
        #     return 0
        else:
            return -1

    @staticmethod
    def get_macd_vote(data: pd.DataFrame, prev_n_days: int, is_selling: bool):
        prev_n_signals = data.loc[prev_n_days, 'MACD-signal'].to_numpy()
        is_selling = 1 if is_selling else -1
        if np.array_equal(prev_n_signals[-2:], np.array([-1, -1])*is_selling):
            return -2
        elif np.array_equal(prev_n_signals[-5:], np.array([-1, -1, 1, 1, 1])*is_selling):
            return 3
        elif np.array_equal(prev_n_signals[-4:], np.array([-1, -1, 1, 1])*is_selling):
            return 2
        elif np.array_equal(prev_n_signals[-3:], np.array([-1, 1, 1])*is_selling):
            return 0
        # elif np.array_equal(prev_n_signals[-2:], np.array([-1, 1])*is_selling):
        #     return 0
        else:
            return -1


def run_strategy_composite(
        manual_strategy, symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int):

    single_symbol_orders = manual_strategy.testPolicy(
        symbol=symbol, start_date=start_date, end_date=end_date, sv=start_value)
    orders = pd.DataFrame(columns=['Symbol', 'Order'])
    for index, date in enumerate(single_symbol_orders.index):
        # if single_symbol_orders.loc[date, 'Order'] == 0:
        #     continue
        orders.loc[date] = [symbol, single_symbol_orders.loc[date, 'Order']]
    port_values = compute_portvals(orders_df=orders, start_val=start_value, 
                                   commission=manual_strategy.commission, impact=manual_strategy.impact)
    port_values = port_values / port_values.iloc[0]
    # episode['Composite'] = port_values

    return port_values, manual_strategy.trade_types


def run_strategy_macd_only(
        episode: pd.DataFrame, symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int):
    learner = ManualStrategy(verbose=False)
    single_symbol_orders = learner.testPolicy(
        symbol=symbol, start_date=start_date, end_date=end_date, sv=start_value,
        use_CCI=False, use_BB=False, use_MACD=True
    )
    orders = pd.DataFrame(columns=['Symbol', 'Order'])
    for index, date in enumerate(single_symbol_orders.index):
        orders.loc[date] = [symbol, single_symbol_orders.loc[date, 'Order']]
    port_values = compute_portvals(orders_df=orders, start_val=start_value, 
                                   commission=learner.commission, impact=learner.impact)
    port_values = port_values / port_values.iloc[0]
    episode['MACD'] = port_values

    return learner.trade_types


def run_strategy_cci_only(
        episode: pd.DataFrame, symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int):
    learner = ManualStrategy(verbose=False)

    single_symbol_orders = learner.testPolicy(
        symbol=symbol, start_date=start_date, end_date=end_date, sv=start_value,
        use_CCI=True, use_BB=False, use_MACD=False
    )
    orders = pd.DataFrame(columns=['Symbol', 'Order'])

    for index, date in enumerate(single_symbol_orders.index):
        orders.loc[date] = [symbol, single_symbol_orders.loc[date, 'Order']]

    port_values = compute_portvals(orders_df=orders, start_val=start_value, 
                                   commission=learner.commission, impact=learner.impact)
    port_values = port_values / port_values.iloc[0]
    episode['CCI'] = port_values

    return learner.trade_types


def run_strategy_bb_only(
        episode: pd.DataFrame, symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int):
    learner = ManualStrategy(verbose=False)

    single_symbol_orders = learner.testPolicy(
        symbol=symbol, start_date=start_date, end_date=end_date, sv=start_value,
        use_CCI=False, use_BB=True, use_MACD=False
    )
    orders = pd.DataFrame(columns=['Symbol', 'Order'])

    for index, date in enumerate(single_symbol_orders.index):
        orders.loc[date] = [symbol, single_symbol_orders.loc[date, 'Order']]

    port_values = compute_portvals(orders_df=orders, start_val=start_value, 
                                   commission=learner.commission, impact=learner.impact)
    port_values = port_values / port_values.iloc[0]
    episode['BB'] = port_values

    return learner.trade_types


# def run_strategy_tos_only(
#         episode: pd.DataFrame, symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int):
#     df_trades = TOS.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
#     df_trades['Symbol'] = symbol
#     df_trades = df_trades[['Symbol', 'Order']]
#     optimal_values = compute_portvals(orders_df=df_trades, start_val=start_value)
#     optimal_values = optimal_values / optimal_values.iloc[0]
#     episode['TOS'] = optimal_values
#
#     return df_trades


def test_and_plot_strategy(
        symbol: str, start_date: dt.datetime, end_date: dt.datetime, start_value: int, charts_prefix: str):
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices = prices / prices.iloc[0]
    episode_portvals = prices.copy()

    # indicators.plot_macd(prices)
    # indicators.plot_commodity_channel_index(prices, span_days=20)
    # Benchmark
    benchmark_orders = pd.DataFrame(index=prices.index, columns=['Symbol', 'Order']).fillna(0)
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.loc[benchmark_orders.index[0]] = (symbol, 1000)
    benchmark_portvals = compute_portvals(orders_df=benchmark_orders, start_val=start_value)
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]
    episode_portvals['Benchmark'] = benchmark_portvals
    #
    # tos_orders = run_strategy_tos_only(
    #     episode=episode_portvals, symbol=symbol, start_date=start_date, end_date=end_date, start_value=start_value)
    # plot_episode_set([
    #     # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
    #     Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
    #     Episode(episode_portvals[['TOS']], f"{symbol} TOS", '#FF0000', tos_orders)],
    #     f'{charts_prefix+" "}Theoretical Optimal Strategy for {symbol}')

    bb_orders = run_strategy_bb_only(
        episode=episode_portvals, symbol=symbol, start_date=start_date, end_date=end_date, start_value=start_value)
    # plot_episode_set([
    #     # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
    #     Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
    #      # Episode(episode_portvals[['TOS']], f"{symbol} TOS"),
    #      Episode(episode_portvals[['BB']], f"{symbol} BB Strategy", '#FF0000', bb_orders),
    #      ],
    #     f'{charts_prefix+" "}BB Only Strategy for {symbol}')

    cci_orders = run_strategy_cci_only(
        episode=episode_portvals, symbol=symbol, start_date=start_date, end_date=end_date, start_value=start_value)
    # plot_episode_set([
    #     # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
    #     Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
    #      # Episode(episode_portvals[['TOS']], f"{symbol} TOS"),
    #      Episode(episode_portvals[['CCI']], f"{symbol} CCI Strategy", '#FF0000', cci_orders),
    #      ],
    #     f'{charts_prefix+" "}CCI Only Strategy for {symbol}')

    macd_orders = run_strategy_macd_only(
        episode=episode_portvals, symbol=symbol, start_date=start_date, end_date=end_date, start_value=start_value)
    # plot_episode_set([
    #     # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
    #     Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
    #      # Episode(episode_portvals[['TOS']], f"{symbol} TOS"),
    #      Episode(episode_portvals[['MACD']], f"{symbol} MACD Strategy", '#FF0000', macd_orders),
    #      ],
    #     f'{charts_prefix+" "}MACD Only Strategy for {symbol}')
    #
    manual_strategy = ManualStrategy(
        verbose=False,
    )
    episode_portvals['Composite'], composite_orders = run_strategy_composite(
        manual_strategy=manual_strategy, symbol=symbol,
        start_date=start_date, end_date=end_date,
        start_value=start_value)
    plot_episode_set([
        # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
        Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
        # Episode(episode_portvals[['TOS']], f"{symbol} TOS"),
        Episode(episode_portvals[['Composite']], f"{symbol} Manual Strategy", '#FF0000', composite_orders),
    ],
        f'{charts_prefix+" "}Manual Composite Strategy for {symbol}')
    # # #
    # Overlay plot
    plot_episode_set([
        # Episode(episode_portvals[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
        Episode(episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
        # [Episode(episode_portvals[[symbol]], f"{symbol} Benchmark"),
         # Episode(episode_portvals[['TOS']], f"{symbol} TOS"),
         Episode(episode_portvals[['CCI']], f"{symbol} CCI Strategy", '#FF0000', cci_orders),
         Episode(episode_portvals[['BB']], f"{symbol} BB Strategy", '#33cc33', bb_orders),
         Episode(episode_portvals[['MACD']], f"{symbol} MACD Strategy", '#FF9933', macd_orders),
         ],
        f'Manual Separate Strategies for {symbol}')
    #
    # # Normalized Overlay plot
    # episode_portvals['CCI-norm'] = episode_portvals['CCI']/episode_portvals['Benchmark']
    # episode_portvals['BB-norm'] = episode_portvals['BB']/episode_portvals['Benchmark']
    # episode_portvals['MACD-norm'] = episode_portvals['MACD']/episode_portvals['Benchmark']
    # episode_portvals['TOS-norm'] = episode_portvals['TOS']/episode_portvals['Benchmark']
    #
    # plot_episode_set(
    #     [Episode(episode_portvals[['Benchmark']]/episode_portvals[['Benchmark']], f"{symbol} Benchmark", '#7A19D9'),
    #      # Episode(episode_portvals[['TOS-norm']], f"{symbol} TOS"),
    #      Episode(episode_portvals[['CCI-norm']], f"{symbol} CCI Strategy", '#00FF00'),
    #      Episode(episode_portvals[['BB-norm']], f"{symbol} BB Strategy", '#0000FF'),
    #      Episode(episode_portvals[['MACD-norm']], f"{symbol} MACD Strategy", '#00FFFF'),
    #      ],
    #     f'Normalized Separate Strategies for {symbol}')

    performance_stats = pd.DataFrame(
        index=['Cumulative Returns', 'Daily Return Mean', 'Daily Return STDEV', 'Sharpe Ratio'])
    performance_stats[f'{charts_prefix+" "}B'] = portfolio_stats(episode_portvals['Benchmark'])
    performance_stats[f'{charts_prefix+" "}S'] = portfolio_stats(episode_portvals['Composite'])
    return performance_stats

def test_manual_strategy():
    seed = 1234567890
    random.seed(seed)

    performance_stats = pd.DataFrame(
        index=['Cumulative Returns', 'Daily Return Mean', 'Daily Return STDEV', 'Sharpe Ratio'])

    symbol = "JPM"
    start_value = 100000
    in_sample_start_date = dt.datetime(2008, 1, 1)
    in_sample_end_date: datetime = dt.datetime(2009, 12, 31)
    stats = test_and_plot_strategy(symbol=symbol,
                           start_date=in_sample_start_date, end_date=in_sample_end_date,
                           start_value=start_value, charts_prefix="In-Sample")
    performance_stats = pd.concat([performance_stats, stats], axis=1)

    out_sample_start_date = dt.datetime(2010, 1, 1)
    out_sample_end_date = dt.datetime(2011, 12, 31)

    stats = test_and_plot_strategy(symbol=symbol,
                           start_date=out_sample_start_date, end_date=out_sample_end_date,
                           start_value=start_value, charts_prefix="Out-Sample")
    performance_stats = pd.concat([performance_stats, stats], axis=1)

    print(performance_stats)


if __name__ == "__main__":
    test_manual_strategy()
