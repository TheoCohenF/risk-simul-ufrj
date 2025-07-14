import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import montecarlo as mc

def test_constant_expected_return():
    print("##### Testing Monte Carlo CER #####")
    cer = mc.ConstantExpectedReturn("AAPL", "1y")
    cer.run(num_simulations=10_000, num_days=30)
    historical_data = cer.get_historical_data()
    print("Historical Data:\n", historical_data)
    future_prices = cer.get_future_prices()
    print("Future Prices:\n", future_prices)
    cer.plot()
    return

def test():
    test_constant_expected_return()
    return

if __name__ == "__main__":
    test()
