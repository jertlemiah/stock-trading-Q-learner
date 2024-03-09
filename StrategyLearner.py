"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Jeremiah Plauche	  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: jplauch3		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903051398	  	   		  		 		  		  		    	 		 		   		 		  
"""
import time
import numpy as np
# import datetime as dt
import random
# import pandas as pd
import util as ut
import QLearner as ql
import marketsimcode as market_sim
from ManualStrategy import *
import indicators as ind
import itertools


class StrategyLearner(object):
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

        holdings_state_options = [0, 1, 2]
        bb_state_options = [0, 1, 2]
        macd_state_options = [0, 1]
        cci_state_options = [0, 1, 2]
        # permutation = itertools.permutations()
        self.possible_states = list(itertools.product(holdings_state_options,
                                                      bb_state_options,
                                                      macd_state_options,
                                                      cci_state_options))

        self.learner = ql.QLearner(
            num_states=len(self.possible_states),
            # num_states=100,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.75,
            radr=0.99,
            # radr=0.999,
            dyna=0,
            verbose=False,
        )

        self.trade_types = pd.DataFrame

        return

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol: str,
            sd: dt.datetime,
            ed: dt.datetime,
            sv: int,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  		 		  		  		    	 		 		   		 		  

        :param symbol: The stock symbol to train on
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        """

        # add your code to do learning here  		  	   		  		 		  		  		    	 		 		   		 		  

        # # example usage of the old backward compatible util function
        syms = [symbol]
        trading_dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # if self.verbose:
        #     print(prices)

        # example use with new colname  		  	   		  		 		  		  		    	 		 		   		 		  
        volume_all = ut.get_data(
            syms, trading_dates, colname="Volume"
        )  # automatically adds SPY  		  	   		  		 		  		  		    	 		 		   		 		  
        volume = volume_all[syms]  # only portfolio symbols  		  	   		  		 		  		  		    	 		 		   		 		  
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		  		 		  		  		    	 		 		   		 		  
        # if self.verbose:
        #     print(volume)

        prev_cumulative_return = -1
        curr_cumulative_return = 0
        count = 0

        while count < 3 or np.absolute(prev_cumulative_return - curr_cumulative_return) > 0.001 and count < 10000:
            episode_begins = time.perf_counter()
            prev_cumulative_return = curr_cumulative_return
            tic = time.perf_counter()
            trades = self.run_episode(symbol, start_date=sd, end_date=ed, sv=sv)
            toc = time.perf_counter()
            if self.verbose:
                print(f"time to trade: {toc - tic:0.4f}")

            orders = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order'])
            orders['Symbol'] = symbol
            orders['Order'] = trades
            # orders = pd.DataFrame(columns=['Symbol', 'Order'])
            # for index, date in enumerate(trades.index):
            #     # if single_symbol_orders.loc[date, 'Order'] == 0:
            #     #     continue
            #     orders.loc[date] = [symbol, trades.loc[date, 'Order']]
            tic = time.perf_counter()
            portvals = market_sim.compute_portvals(
                orders_df=orders,
                start_val=sv,
                commission=self.commission,
                impact=self.impact,
            )
            toc = time.perf_counter()
            if self.verbose:
                print(f"time to compute portvals: {toc - tic:0.4f}")

            # curr_cumulative_return = portvals.iloc[-1][0]/portvals.iloc[0][0] - 1
            # curr_cumulative_return = portvals.iloc[-1] / portvals.iloc[0] - 1
            curr_cumulative_return, _, _, _ = portfolio_stats(portvals)
            episode_ends = time.perf_counter()
            if self.verbose:
                print(
                    f"episode {count}, cumulative  returns {curr_cumulative_return}, time {episode_ends - episode_begins:0.4f}")
            count += 1
            # avg_daily_ret, sharpe_ratio = marketsimcode.get_stats(portvals)

    def get_reward(self, price_today, price_yesterday, holding):
        if holding == 0:
            return 0
        impact_val = (1.0 + self.impact) if holding < 0 else (1.0 - self.impact)
        daily_return = (price_today*impact_val / price_yesterday) - 1.

        return (holding / 1000) * daily_return

    def run_episode(self,
                    symbol="JPM",
                    start_date=dt.datetime(2008, 1, 1),
                    end_date=dt.datetime(2009, 1, 1),
                    sv=100000,
                    ):
        raw_prices = get_data([symbol], pd.date_range(start_date, end_date))
        prices = market_sim.get_normalized_prices(symbols=[symbol],
                                                  start_date=start_date,
                                                  end_date=end_date)
        indicators = ind.get_normalized_indicators(prices=prices)
        partial_states = discretize_indicators_into_states(indicators)

        # orders = pd.DataFrame(index=prices.index, columns=['Order', 'Date', 'Symbol', 'Shares'])
        orders = pd.DataFrame(index=prices.index, columns=['Order'])
        holdings = 0
        yesterdays_holdings = 0
        state = self.generate_state(holdings, partial_states.iloc[0][0])
        action = self.learner.querysetstate(state)
        current_cash = sv
        previous_cash = sv
        state_prime = state

        for i, today in enumerate(prices.index):
            yesterday = prices.index[i - 1] if i > 0 else today
            # self.current_shares = 0

            state_prime = self.generate_state(holdings, partial_states.loc[today][0])

            reward = self.get_reward(prices.loc[today, symbol], prices.loc[yesterday, symbol], holdings)

            action = self.learner.query(state_prime, reward)

            # Actions
            #   0: SHORT
            #   1: CASH
            #   2: LONG
            #   (hold)
            if action == 0 and (holdings == 1000 or holdings == 0):
                trade_shares = -1000 - holdings
            elif action == 1 and abs(holdings) == 1000:
                trade_shares = -holdings
            elif action == 2 and (holdings == -1000 or holdings == 0):
                trade_shares = 1000 - holdings
            else:
                trade_shares = 0

            yesterdays_holdings = holdings
            holdings += trade_shares
            orders.loc[today, 'Order'] = trade_shares

            impact_val = (1 + self.impact) if trade_shares > 0 else (1 - self.impact)

            previous_cash = current_cash
            current_cash += -raw_prices.loc[today, symbol] * impact_val * trade_shares
            # state_prime = self.generate_state(holdings, partial_states.loc[today][0])

            # action = learner.query(state, reward)
        return orders

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol: str,
            sd: dt.datetime,
            ed: dt.datetime,
            sv: int,
    ):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        Tests your learner using data outside the training data

        :param symbol: The stock symbol that you trained on
        :type symbol: str  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        prices = market_sim.get_normalized_prices(symbols=[symbol],
                                                  start_date=sd,
                                                  end_date=ed)
        indicators = ind.get_normalized_indicators(prices=prices)
        partial_states = discretize_indicators_into_states(indicators)
        self.trade_types = pd.DataFrame(index=prices.index, columns=['OrderType', 'Shares'])

        # orders = pd.DataFrame(index=prices.index, columns=['Order', 'Date', 'Symbol', 'Shares'])
        orders = pd.DataFrame(index=prices.index, columns=['Shares'])
        orders.values[0, :] = 1000  # add a BUY at the start
        holding = 1000
        state = self.generate_state(holding, partial_states.iloc[0][0])
        action = self.learner.querysetstate(state)

        for i, today in enumerate(prices.index):
            if i == 0:
                continue
            yesterday = prices.index[i - 1] if i > 0 else today

            state_prime = self.generate_state(holding, partial_states.loc[today][0])
            # print(f"Loop {i}, state {state_prime}")
            action = self.learner.querysetstate(state_prime)

            # Actions
            #   0: SHORT
            #   1: CASH
            #   2: LONG
            #   (hold)
            if action == 0 and (holding == 1000 or holding == 0):
                trade_shares = -1000 - holding
            elif action == 1 and abs(holding) == 1000:
                trade_shares = -holding
            elif action == 2 and (holding == -1000 or holding == 0):
                trade_shares = 1000 - holding
            else:
                trade_shares = 0

            holding += trade_shares
            orders.loc[today, 'Shares'] = trade_shares
            self.trade_types.loc[today] = (action, trade_shares)

            # action = learner.query(state, reward)
        return orders
        # return trades

    def generate_state(self, holdings, partial_state):
        # def generate_state(self, holdings, bb, macd, cci):
        # Total states is 3*3*2*5=90

        # Holdings only has 3 states:
        #   0: when holdings == -1000
        #   1: when holdings == 0
        #   2: when holdings == 1000
        if holdings == -1000:
            holdings = 0
        elif holdings == 0:
            holdings = 1
        elif holdings == 1000:
            holdings = 2

        (bb, macd, cci) = partial_state

        # return int(str(holdings)+str(partial_state))
        return self.possible_states.index((holdings, bb, macd, cci))


def discretize_indicators_into_states(indicators: pd.DataFrame):
    # Bollinger Bands have 3 states:
    #   0: when EMA < lower band
    #   1: when EMA > lower band and EMA < upper band
    #   2: when EMA > upper band
    bb = (indicators[['BB-signal']] + 1)

    # MACD has 2 states:
    #   0: when MACD <= signal line
    #   1: when MACD > signal line
    macd = (indicators[['MACD-signal']]).replace(-1, 0)

    # CCI has 5 ish states:
    #   0: CCI <= -2
    #   1: CCI == -1
    #   2: CCI == 0
    #   3: CCI == 1
    #   4. CCI >= 2
    cci = indicators[['CCI-signal']]
    # conditions = [
    #     cci <= -2,
    #     cci == -1,
    #     cci == 0,
    #     cci == 1,
    #     cci >= 2
    # ]
    # choices = [0, 1, 2, 3, 4]
    conditions = [
        cci <= -1,
        cci == 0,
        cci >= 1,
    ]
    choices = [0, 1, 2]
    cci = np.select(conditions, choices, default=0)
    cci = pd.DataFrame(cci, index=indicators[['CCI-signal']].index, columns=['CCI-signal'])

    # return (bb.iloc[:, 0].astype(str) + macd.iloc[:, 0].astype(str) + cci.iloc[:, 0].astype(str)).astype(int)
    partial_states = pd.DataFrame(index=indicators.index)
    partial_states["States"] = tuple(zip(bb.iloc[:, 0].to_list(), macd.iloc[:, 0].to_list(), cci.iloc[:, 0].to_list()))
    return partial_states


def test_learner():
    random.seed("jplauch3")
    random.seed(1481090000)
    # symbol = "JPM"
    symbol = "ML4T-220"
    # symbol = "SINE_FAST_NOISE"
    start_value = 100000
    commission = 9.95
    impact = 0
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    learner = StrategyLearner(impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
    df_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
    orders = pd.DataFrame(index=df_trades.index, columns=['Symbol', 'Order'])
    orders['Symbol'] = symbol
    orders['Order'] = df_trades
    port_values = market_sim.compute_portvals(
        orders_df=orders,
        start_val=start_value,
        commission=learner.commission,
        impact=learner.impact)
    port_values = port_values / port_values.iloc[0]

    # final_cumulative_return = port_values.iloc[-1] / port_values.iloc[0] - 1
    final_cumulative_return, _, _, _ = portfolio_stats(port_values)
    print(f"final_cumulative_return {final_cumulative_return}")

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices_norm = prices / prices.iloc[0]
    episode = prices_norm.copy()
    episode['Learner'] = port_values

    # indicators.plot_macd(prices_norm)
    # indicators.plot_commodity_channel_index(prices_norm, span_days=20)
    # Benchmark
    benchmark_orders = pd.DataFrame(index=prices.index, columns=['Symbol', 'Order']).fillna(0)
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.loc[benchmark_orders.index[0]] = (symbol, 1000)
    optimal_values = compute_portvals(orders_df=benchmark_orders, start_val=start_value,
                                      commission=learner.commission, impact=learner.impact)
    optimal_values = optimal_values / optimal_values.iloc[0]
    episode['Benchmark'] = optimal_values

    plot_episode_set([
        # Episode(episode[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
        Episode(episode[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
        # Episode(episode[['TOS']], f"{symbol} TOS"),
        Episode(episode[['Learner']], f"{symbol} Strategy QLearner", '#FF0000', learner.trade_types),
    ],
        f'Strategy Learner for {symbol}')
    return


if __name__ == "__main__":
    test_learner()
