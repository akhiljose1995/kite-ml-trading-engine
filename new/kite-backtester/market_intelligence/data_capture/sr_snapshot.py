import pandas as pd
from typing import Dict, List


class SupportResistanceSnapshot:
    """
    Extracts support and resistance zones using swing detection + clustering.
    """

    def __init__(self, detector, zone_cluster):
        self.detector = detector
        self.zone_cluster = zone_cluster
        self.zones = None

    def capture(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> Dict[str, List[dict]]:

        if df is None or df.empty:
            return {"above": [], "below": []}

        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        result_df = self.detector.detect(df)

        # ---------------------------------------------
        # Filter only swing points
        # ---------------------------------------------
        swing_df = result_df[
            (result_df["swing_high"]) | (result_df["swing_low"])
        ].copy()

        # Add helper column for readability
        swing_df["swing_type"] = None
        swing_df.loc[swing_df["swing_high"], "swing_type"] = "SWING_HIGH"
        swing_df.loc[swing_df["swing_low"], "swing_type"] = "SWING_LOW"

        zones = self.zone_cluster.cluster(swing_df)
        self.zones = zones

        # Return last zone for above and first zone for below
        sr_zones = self.get_above_and_below_zones(current_price)
        
        #print("Above zones:", sr_zones["above_price"])
        #print("Below zones:", sr_zones["below_price"])
        return sr_zones

    def get_above_and_below_zones(self, current_price) -> Dict[str, List[dict]]:
        if self.zones is None:
            return {"above": [], "below": []}

        above, below = [], []

        for _, row in self.zones.iterrows():
            zone = row.to_dict()
            level = zone.get("zone_center")

            if level is None:
                continue

            if level > current_price:
                above.append(zone)
            else:
                below.append(zone)

        # Sort above and below in ascending order of zone_center
        above.sort(key=lambda x: x.get("zone_center", float("inf")))
        below.sort(key=lambda x: x.get("zone_center", float("-inf")))
        # Return last zone for above and first zone for below
        above_zone = above[:2] if above else []
        below_zone = below[-2:] if below else []
        #print("Above zones:", above_zone)
        #print("Below zones:", below_zone)
        return {
            "above_price": above_zone,
            "below_price": below_zone
        }