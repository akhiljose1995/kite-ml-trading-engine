import pytest
import datetime
from data.kite_loader import KiteDataLoader
from kiteconnect import KiteConnect
import config
import ta
from strategy.expiry_day_gamma_blasts import ExpiryDayGammaBlast

@pytest.fixture(scope="module")
def kite_obj():
    """
    Fixture to create and return a KiteConnect object.
    """
    kite_loader = KiteDataLoader(
        api_key=config.API_KEY,
        access_token=config.ACCESS_TOKEN,
        request_token=config.REQUEST_TOKEN,
        api_secret=config.API_SECRET
    )
    kite_object = kite_loader.kite_object()
    return kite_object

def test_expiry_day_gamma_blasts(kite_obj):
    """
    Test function to validate expiry day gamma blasts for a specific index.
    """
    index = "sensex"  # Change to "banknifty" or "sensex" as needed
    save_path = f"data/{index}_gamma_blasts.csv"
    # Process expiry days and generate CSV
    if index.lower() == "nifty50":
        #start_date='2019-02-01'
        start_date='2024-01-01'
        price_diff_threshold = 50
        start_time = "14:00"
    elif index.lower() == "banknifty":
        start_date='2019-01-01'
        price_diff_threshold = 150
        start_time = "13:30"
    elif index.lower() == "sensex":
        #start_date='2018-10-26'
        start_date='2024-01-01'
        price_diff_threshold = 200
        start_time = "14:30"

    # Initialize the gamma blast strategy
    gamma = ExpiryDayGammaBlast(kite=kite_obj, start_date=start_date, index=index)
    
    gamma.process_expiry_days(save_path=save_path, start_time=start_time, limit=None, price_diff_threshold=price_diff_threshold)

    # Plot summary of gamma moves
    gamma.plot_blast_summary(csv_path=save_path)

    # Show top moves in a styled table
    gamma.display_blast_table(csv_path=save_path, max_rows=20)

    """ # Process each expiry day
    for expiry in expiry_days:
        print(f"Processing expiry day: {expiry}")
        gamma.fetch_intraday_data(expiry)
    
    # Save results to CSV
    try:
        gamma.save_gamma_blasts_to_csv(save_path="data/nifty_gamma_blasts.csv")
        print("Gamma blasts saved successfully.") """



