import pytest
import datetime
from data.kite_loader import KiteDataLoader
from strategy.ema_crossover import EMACrossoverStrategy
from portfolio.portfolio import Portfolio
from report.visualizer import EquityCurvePlot
import config
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def get_real_equity_data():
    # Load real data
    loader = KiteDataLoader(
        api_key=config.API_KEY,
        access_token=config.ACCESS_TOKEN,
        request_token=config.REQUEST_TOKEN,
        api_secret=config.API_SECRET
    )
    
    df = loader.get_data(
        instrument_token=config.INSTRUMENT_TOKEN,
        from_date=datetime.datetime(2025, 5, 16),
        to_date=datetime.datetime(2025, 5, 17),
        interval="15minute"
    )

    # Run backtest with real data
    strategy = EMACrossoverStrategy(short_window=5, long_window=20)
    signals = strategy.generate_signals(df)

    portfolio = Portfolio(initial_capital=100000)

    for i, row in signals.iterrows():
        timestamp = row.name
        price = row['close']
        signal = row['signal']
        if signal == 1:
            portfolio.buy(price, timestamp)
        elif signal == -1:
            portfolio.sell(price, timestamp)
        portfolio.record_equity(price)

    return portfolio.get_equity_curve(), signals.index.tolist()


def test_plot_equity_curve_real():
    equity_curve, timestamps = get_real_equity_data()
    plotter = EquityCurvePlot(equity_curve, timestamps)

    # Just ensure no exception is raised during plotting
    try:
        plotter.plot(show=True)  # Suppress display in test
    except Exception as e:
        pytest.fail(f"Plotting with real data failed: {e}")

def test_save_equity_curve_real(tmp_path):
    equity_curve, timestamps = get_real_equity_data()
    output_path = tmp_path / "real_equity_curve.png"

    plotter = EquityCurvePlot(equity_curve, timestamps)
    plotter.save(path=str(output_path))

    assert output_path.exists(), "Real equity curve plot was not saved"
