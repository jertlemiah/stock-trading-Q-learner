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


def run_experiment_2(symbol="JPM"):
    start_value = 100000
    commission = 9.95
    # commission = 0
    verbose = False

    # in-sample
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # removes SPY, which is automatically added
    prices_norm = prices / prices.iloc[0]
    episode = prices_norm.copy()

    impact = 0
    learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

    df_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
    learner0_orders = pd.DataFrame(index=df_trades.index, columns=['Symbol', 'Order'])
    learner0_orders['Symbol'] = symbol
    learner0_orders['Order'] = df_trades
    port_values = market_sim.compute_portvals(
        orders_df=learner0_orders,
        start_val=start_value,
        commission=commission,
        impact=impact)
    port_values = port_values / port_values.iloc[0]
    episode['Learner0'] = port_values

    impact = 0.005
    learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

    df_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
    learner0_005_orders = pd.DataFrame(index=df_trades.index, columns=['Symbol', 'Order'])
    learner0_005_orders['Symbol'] = symbol
    learner0_005_orders['Order'] = df_trades
    port_values = market_sim.compute_portvals(
        orders_df=learner0_orders,
        start_val=start_value,
        commission=commission,
        impact=impact)
    port_values = port_values / port_values.iloc[0]
    episode['Learner0.005'] = port_values

    impact = 0.01
    learner = sl.StrategyLearner(verbose=verbose, commission=commission, impact=impact)
    learner.add_evidence(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)

    df_trades = learner.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=start_value)
    learner0_01_orders = pd.DataFrame(index=df_trades.index, columns=['Symbol', 'Order'])
    learner0_01_orders['Symbol'] = symbol
    learner0_01_orders['Order'] = df_trades
    port_values = market_sim.compute_portvals(
        orders_df=learner0_orders,
        start_val=start_value,
        commission=commission,
        impact=impact)
    port_values = port_values / port_values.iloc[0]
    episode['Learner0.01'] = port_values

    plot_episode_set([
        # Episode(episode[[symbol]], f"{symbol} Prices", '#ff66ff', benchmark_orders),
        Episode(episode[['Learner0']], f"{symbol} Learner: Impact 0", '#7A19D9', learner0_orders),
        # Episode(episode[['TOS']], f"{symbol} TOS"),
        Episode(episode[['Learner0.005']], f"{symbol} Learner: Impact 0.005", '#FF0000', learner0_005_orders),
        Episode(episode[['Learner0.01']], f"{symbol} Learner: Impact 0.01", '#00FF00', learner0_01_orders),
    ],
        f'Experiment 2 for {symbol}: Portvals')



if __name__ == "__main__":
    random.seed("jplauch3")
    run_experiment_2("JPM")
