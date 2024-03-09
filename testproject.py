#    Code initializing/running all necessary files for the report.
#    NOTE: You will have to create the contents of this file yourself.

# import StrategyLearner as sl
import ManualStrategy as ms
import experiment1
import experiment2
# from strategy_evaluation import ManualStrategy, experiment1


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jplauch3"


if __name__ == "__main__":
    print("")
    ms.test_manual_strategy()
    experiment1.run_experiment_1(symbol="JPM")
    experiment2.run_experiment_2(symbol="JPM")

