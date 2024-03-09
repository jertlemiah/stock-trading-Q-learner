# import numpy as np
# import datetime as dt
# import random
# import pandas as pd
# import util as ut
# import QLearner as ql
import marketsimcode as market_sim
from ManualStrategy import *
# import indicators as ind
# import itertools
import StrategyLearner as sl


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


def compare_strategies(
        learner, manual_strategy, symbol: str,
        start_date: dt.datetime, end_date: dt.datetime,
        start_value: int, chart_prefix: str
):
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

    final_cumulative_return = port_values.iloc[-1] / port_values.iloc[0] - 1
    print(f"final_cumulative_return {final_cumulative_return}")

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices_norm = prices / prices.iloc[0]
    episode = prices_norm.copy()
    episode['Learner'] = port_values

    # Benchmark
    benchmark_orders = pd.DataFrame(index=prices.index, columns=['Symbol', 'Order']).fillna(0)
    benchmark_orders['Symbol'] = symbol
    benchmark_orders.loc[benchmark_orders.index[0]] = (symbol, 1000)
    optimal_values = compute_portvals(orders_df=benchmark_orders, start_val=start_value,
                                      commission=learner.commission, impact=learner.impact)
    optimal_values = optimal_values / optimal_values.iloc[0]
    episode['Benchmark'] = optimal_values

    portvals, composite_orders = run_strategy_composite(
        manual_strategy=manual_strategy, symbol=symbol,
        start_date=start_date, end_date=end_date, start_value=start_value)
    episode['Composite'] = portvals

    plot_episode_set([
        # Episode(episode[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
        Episode(episode[['Benchmark']], f"{symbol} Benchmark", '#7A19D9', benchmark_orders),
        # Episode(episode[['TOS']], f"{symbol} TOS"),
        Episode(episode[['Composite']], f"{symbol} Manual Strategy", '#FF0000', composite_orders),
        Episode(episode[['Learner']], f"{symbol} Strategy QLearner", '#00FF00', orders),
        ],
        f'{chart_prefix+" "}Experiment 1 for {symbol}')

    return


def run_experiment_1(symbol="JPM"):
    start_value = 100000
    commission = 9.95
    # commission = 0
    impact = 0.005
    # impact = 0
    verbose = False
    # in-sample
    in_sample_start_date = dt.datetime(2008, 1, 1)
    in_sample_end_date = dt.datetime(2009, 12, 31)

    learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol=symbol, sd=in_sample_start_date, ed=in_sample_end_date, sv=start_value)

    manual_strategy = ManualStrategy(verbose=verbose, commission=commission, impact=impact)

    compare_strategies(learner=learner, manual_strategy=manual_strategy, symbol=symbol,
                       start_date=in_sample_start_date, end_date=in_sample_end_date,
                       start_value=start_value, chart_prefix="In-Sample")

    # out-sample
    out_sample_start_date = dt.datetime(2010, 1, 1)
    out_sample_end_date = dt.datetime(2011, 12, 31)

    compare_strategies(learner=learner, manual_strategy=manual_strategy, symbol=symbol,
                       start_date=out_sample_start_date, end_date=out_sample_end_date,
                       start_value=start_value, chart_prefix="Out-Sample")


if __name__ == "__main__":
    seed = 1481090000
    np.random.seed(seed)
    random.seed(seed)
    # random.seed("jplauch3")
    run_experiment_1("JPM")
    # run_experiment_1("ML4T-220")
    # run_experiment_1("AAPL")
    # run_experiment_1("SINE_FAST_NOISE")
