import pandas as pd
import os
import traceback
from datetime import datetime, timedelta
import pytz
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from config import option_chain_url_dict

class OptionChainTracker:
    def __init__(self, driver, index_symbol="nifty50"):
        try:
            self.driver = driver
            self.index_symbol = index_symbol
            self.url = option_chain_url_dict.get(index_symbol)
            self.csv_path = f"{index_symbol}_oi_browser.csv"
            self.strike_list = []
            self.df_dict = {strike: pd.DataFrame(columns=["date", "Call OI", "Put OI"]) for strike in self.strike_list}
            self.prev_oi = {strike: {"call": None, "put": None} for strike in self.strike_list}
            self._load_existing_data()
        except Exception:
            print("Error in __init__:\n", traceback.format_exc())

    def _load_existing_data(self):
        try:
            if os.path.exists(self.csv_path):
                full_df = pd.read_csv(self.csv_path, parse_dates=["date"])
                for strike in self.strike_list:
                    strike_df = full_df[full_df["strike"] == strike][["date", "Call OI", "Put OI"]]
                    self.df_dict[strike] = strike_df.reset_index(drop=True)
        except Exception:
            print("Error in _load_existing_data:\n", traceback.format_exc())
    
    def fetch_rows(self):
        try:
            self.driver.get(self.url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
            )
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            return soup.select("table tr")
        except Exception:
            print("Error in fetch_rows:\n", traceback.format_exc())
            return []

    def parse_snapshot(self, rows):
        try:
            snapshot = {}
            error_strike = ""
            for row in rows:
                call_index = 0
                put_index = 7
                strike_index = 4
                cells = row.text.split()
                #print(f"Row cells: {cells}")
                if len(cells) >= 8:
                    if 'P' in cells[strike_index]:
                        strike_index += 1
                        put_index += 1
                        
                    strike = cells[strike_index]
                    if strike in self.strike_list:
                        if 'P' in cells[strike_index+1]:
                            put_index += 1
                        #print(f"Row cells: {cells}")
                        error_strike = strike
                        snapshot[strike] = {
                            "call": float(cells[call_index]),
                            "put": float(cells[put_index])
                        }
            #print(f"Parsed snapshot: {snapshot}")
            return snapshot
        except Exception:
            print("Error in parse_snapshot:\n", traceback.format_exc())
            return {}

    def has_changed(self, snapshot):
        try:
            #print(f"strike_list: {self.strike_list}")
            for strike in self.strike_list:
                if snapshot.get(strike):
                    if (snapshot[strike]["call"] != self.prev_oi[strike]["call"] or
                        snapshot[strike]["put"] != self.prev_oi[strike]["put"]):
                        return True
            return False
        except Exception:
            print("Error in has_changed:\n", traceback.format_exc())
            return False

    def update_dataframes(self, snapshot):
        try:
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now().replace(microsecond=0)
            now = ist.localize(now)
            updated_rows = []

            for strike in self.strike_list:
                if strike in snapshot:
                    call_oi = snapshot[strike]["call"]
                    put_oi = snapshot[strike]["put"]
                    new_row = {"date": now, "Call OI": call_oi, "Put OI": put_oi}

                    self.df_dict[strike] = self.df_dict[strike][self.df_dict[strike]["date"] != now]
                    self.df_dict[strike] = pd.concat([
                        self.df_dict[strike],
                        pd.DataFrame([new_row])
                    ], ignore_index=True)

                    updated_rows.append({
                        "date": now,
                        "strike": strike,
                        "Call OI": call_oi,
                        "Put OI": put_oi
                    })

                    self.prev_oi[strike] = {"call": call_oi, "put": put_oi}

            if updated_rows:
                full_df = pd.DataFrame(updated_rows)
                if os.path.exists(self.csv_path):
                    existing_df = pd.read_csv(self.csv_path, parse_dates=["date"])
                    existing_df = existing_df[~existing_df.set_index(["date", "strike"]).index.isin(
                        full_df.set_index(["date", "strike"]).index)]
                    full_df = pd.concat([existing_df, full_df], ignore_index=True)
                full_df.to_csv(self.csv_path, index=False)
        except Exception:
            print("Error in update_dataframes:\n", traceback.format_exc())

    def run_once(self, contract_details):
        try:
            self.strike_list = [str(contract_details[k]['strike']) for k in [
                "itm2_pe", "itm1_pe", "atm_pe", "otm1_pe", "otm2_pe"
            ]]
            for strike in self.strike_list:
                if strike not in self.df_dict:
                    self.df_dict[strike] = pd.DataFrame(columns=["date", "Call OI", "Put OI"])
                    self.prev_oi[strike] = {"call": None, "put": None}
            rows = self.fetch_rows()
            snapshot = self.parse_snapshot(rows)
            if self.has_changed(snapshot):
                self.update_dataframes(snapshot)
                return True, self.df_dict
            return False, self.df_dict
        except Exception:
            print("Error in run_once:\n", traceback.format_exc())
            return False, self.df_dict
