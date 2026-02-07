import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import DBSCAN


# -------------------------------------------------
# Configuration
# -------------------------------------------------

@dataclass
class ZoneClusterConfig:
    price_column_high: str = "swing_high_price"
    price_column_low: str = "swing_low_price"

    max_zone_width: float = 120.0      # HARD CAP
    min_samples: int = 2              # For DBSCAN only

    # Strength config
    max_age_days: int = 365 * 5
    max_touch_cap: int = 8

    weight_age: float = 0.65
    weight_touch: float = 0.35
    weight_extreme: float = 0.65
    extreme_decay_scale: float = 50.0


# -------------------------------------------------
# Zone Clustering Engine
# -------------------------------------------------

zone_cluster_config = ZoneClusterConfig()

class ZoneCluster:
    """
    Clusters swing highs and lows into tight horizontal zones.
    Also emits orphan zones for single-touch important levels.
    """

    def __init__(self, config: ZoneClusterConfig = zone_cluster_config):
        self.config = config
        self.eps = self.config.max_zone_width / 2.0
        self.global_ath = None
        self.global_atl = None

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        swing_df = self._extract_swing_points(df)

        if swing_df.empty:
            return pd.DataFrame()

        self.global_ath = swing_df["price"].max()
        self.global_atl = swing_df["price"].min()
        swing_df["cluster_id"] = self._run_dbscan(swing_df["price"].values)

        clustered_zones = self._build_clustered_zones(swing_df)
        orphan_zones = self._build_orphan_zones(swing_df)

        zones_df = pd.concat([clustered_zones, orphan_zones], ignore_index=True)

        return zones_df.sort_values(
            by=["zone_center", "strength_score"],
            ascending=False
        ).reset_index(drop=True)

    # -------------------------------------------------
    # Swing Extraction
    # -------------------------------------------------

    def _extract_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []

        for ts, row in df.iterrows():
            if not pd.isna(row.get(self.config.price_column_high)):
                records.append({"timestamp": ts, "price": float(row[self.config.price_column_high])})

            if not pd.isna(row.get(self.config.price_column_low)):
                records.append({"timestamp": ts, "price": float(row[self.config.price_column_low])})

        return pd.DataFrame(records)

    def _run_dbscan(self, prices: np.ndarray) -> np.ndarray:
        model = DBSCAN(
            eps=self.eps,
            min_samples=self.config.min_samples,
            metric="euclidean"
        )
        return model.fit_predict(prices.reshape(-1, 1))

    # -------------------------------------------------
    # Clustered Zones
    # -------------------------------------------------

    def _build_clustered_zones(self, swing_df: pd.DataFrame) -> pd.DataFrame:
        zones = []
        now = pd.Timestamp.utcnow()

        valid_clusters = [c for c in swing_df["cluster_id"].unique() if c != -1]

        for cid in valid_clusters:
            cluster_df = swing_df[swing_df["cluster_id"] == cid]

            zone_low = cluster_df["price"].min()
            zone_high = cluster_df["price"].max()
            zone_width = zone_high - zone_low

            # Enforce HARD width
            if zone_width > 1.5*self.config.max_zone_width:
                continue

            zone_center = round(float(cluster_df["price"].mean()), 2)
            touch_count = len(cluster_df)

            first_touch = cluster_df["timestamp"].min()
            last_touch = cluster_df["timestamp"].max()
            zone_age_days = (now - first_touch).days

            strength = self._compute_strength(zone_age_days, touch_count, zone_center)

            zones.append({
                "zone_id": f"C{cid}",
                "zone_category": "CLUSTERED",
                "zone_center": zone_center,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "zone_width": zone_width,
                "touch_count": touch_count,
                "zone_age_days": zone_age_days,
                "strength_score": strength,
                "strength_label": self._label_strength(strength),
                "first_touch_time": first_touch,
                "last_touch_time": last_touch,
            })

        return pd.DataFrame(zones)

    # -------------------------------------------------
    # Orphan Zones
    # -------------------------------------------------

    def _build_orphan_zones(self, swing_df: pd.DataFrame) -> pd.DataFrame:
        zones = []
        now = pd.Timestamp.utcnow()

        orphan_df = swing_df[swing_df["cluster_id"] == -1]

        for idx, row in orphan_df.iterrows():
            zone_age_days = (now - row["timestamp"]).days

            strength = self._compute_strength(zone_age_days, touch_count=1, zone_center=row["price"])

            zones.append({
                "zone_id": f"O{idx}",
                "zone_category": "ORPHAN",
                "zone_center": row["price"],
                "zone_low": row["price"],
                "zone_high": row["price"],
                "zone_width": 0.0,
                "touch_count": 1,
                "zone_age_days": zone_age_days,
                "strength_score": strength,
                "strength_label": self._label_strength(strength),
                "first_touch_time": row["timestamp"],
                "last_touch_time": row["timestamp"],
            })

        return pd.DataFrame(zones)
    
    # -------------------------------------------------
    # Extreme Score (for future use, not in current strength)
    # -------------------------------------------------
    def _compute_extreme_score(self, price: float) -> float:
        d_ath = abs(self.global_ath - price)
        d_atl = abs(price - self.global_atl)

        score_ath = np.exp(-d_ath / self.config.extreme_decay_scale)
        score_atl = np.exp(-d_atl / self.config.extreme_decay_scale)

        return max(score_ath, score_atl)

    # -------------------------------------------------
    # Strength Logic
    # -------------------------------------------------

    def _compute_strength(self, zone_age_days: int, touch_count: int, zone_center) -> float:
        capped_age = min(zone_age_days, self.config.max_age_days)
        age_score = np.log1p(capped_age) / np.log1p(self.config.max_age_days)
        extreme_score = self._compute_extreme_score(zone_center)

        capped_touch = min(touch_count, self.config.max_touch_cap)
        touch_score = np.log1p(capped_touch) / np.log1p(self.config.max_touch_cap)

        strength = (
            self.config.weight_age * age_score +
            self.config.weight_touch * touch_score +
            self.config.weight_extreme * extreme_score
        )
        return round(min(strength, 1.0), 4)

    def _label_strength(self, score: float) -> str:
        if score >= 0.75:
            return "VERY_STRONG"
        elif score >= 0.55:
            return "STRONG"
        elif score >= 0.35:
            return "MODERATE"
        else:
            return "WEAK"

"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.cluster import DBSCAN
from datetime import datetime


# -------------------------------------------------
# Configuration
# -------------------------------------------------

@dataclass
class ZoneClusterConfig:
    atr_column: str = "atr"
    price_column_high: str = "swing_high_price"
    price_column_low: str = "swing_low_price"

    eps_atr_multiplier: float = 1.0     # Zone width tolerance
    min_samples: int = 2                # Minimum swings to form a zone

    strength_decay_cap_days: int = 365 * 5  # Cap age weight (5 years)


# -------------------------------------------------
# Zone Clustering
# -------------------------------------------------

class ZoneCluster:"""
"""
    Clusters swing highs/lows into zones using DBSCAN and computes zone metrics.
    """

"""
    def __init__(self, config: ZoneClusterConfig):
        self.config = config

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        swing_df = self._extract_swing_points(df)

        if swing_df.empty:
            return pd.DataFrame()

        eps = self._compute_eps(swing_df)

        clusters = self._run_dbscan(swing_df["price"].values, eps)

        swing_df["cluster_id"] = clusters

        zones = self._build_zones(swing_df)

        return zones

    # -------------------------------------------------
    # Internal Steps
    # -------------------------------------------------

    def _extract_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:"""
"""#Convert swing highs and lows into a unified price series.
    """
"""
        records = []

        for ts, row in df.iterrows():
            if not pd.isna(row.get(self.config.price_column_high)):
                records.append({
                    "timestamp": ts,
                    "price": row[self.config.price_column_high],
                    "type": "HIGH",
                    "atr": row[self.config.atr_column],
                })

            if not pd.isna(row.get(self.config.price_column_low)):
                records.append({
                    "timestamp": ts,
                    "price": row[self.config.price_column_low],
                    "type": "LOW",
                    "atr": row[self.config.atr_column],
                })

        return pd.DataFrame(records)

    def _compute_eps(self, swing_df: pd.DataFrame) -> float:"""
"""
        Compute clustering distance using ATR.
        """
"""
        atr_mean = swing_df["atr"].dropna().mean()
        return atr_mean * self.config.eps_atr_multiplier

    def _run_dbscan(self, prices: np.ndarray, eps: float) -> np.ndarray:"""
"""
        Run 1D DBSCAN clustering on price axis.
        """
"""
        model = DBSCAN(
            eps=eps,
            min_samples=self.config.min_samples,
            metric="euclidean"
        )

        return model.fit_predict(prices.reshape(-1, 1))

    # -------------------------------------------------
    # Zone Construction
    # -------------------------------------------------

    def _build_zones(self, swing_df: pd.DataFrame) -> pd.DataFrame:
        zones = []

        valid_clusters = swing_df["cluster_id"].unique()
        valid_clusters = valid_clusters[valid_clusters != -1]

        now = pd.Timestamp.utcnow()

        for cid in valid_clusters:
            cluster_df = swing_df[swing_df["cluster_id"] == cid]

            zone_low = cluster_df["price"].min()
            zone_high = cluster_df["price"].max()

            # Recency-weighted center
            center = self._compute_weighted_center(cluster_df, now)

            zone_type = self._infer_zone_type(cluster_df)
            strength_score = self._compute_zone_strength(cluster_df, now)
            strength_label = self._label_strength(strength_score)

            zones.append({
                "zone_id": cid,
                "zone_type": zone_type,
                "zone_center": center,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "zone_width": zone_high - zone_low,
                "touch_count": len(cluster_df),
                "strength_score": strength_score,
                "strength_label": strength_label,
                "first_touch_time": cluster_df["timestamp"].min(),
                "last_touch_time": cluster_df["timestamp"].max(),
            })

        return pd.DataFrame(zones)

    # -------------------------------------------------
    # Zone Metrics
    # -------------------------------------------------

    def _compute_weighted_center(self, cluster_df: pd.DataFrame, now: pd.Timestamp) -> float:"""
"""
        Older swings get higher weight (log-scaled).
        """
"""
        ages = (now - cluster_df["timestamp"]).dt.days.clip(
            upper=self.config.strength_decay_cap_days
        )

        weights = np.log1p(ages)
        return np.average(cluster_df["price"], weights=weights)

    def _compute_zone_strength(self, cluster_df: pd.DataFrame, now: pd.Timestamp) -> float:"""
"""
        Strength based on recency-weighted touch count.
        """
"""
        ages = (now - cluster_df["timestamp"]).dt.days.clip(
            upper=self.config.strength_decay_cap_days
        )

        weights = np.log1p(ages)

        raw_strength = weights.sum()

        # Normalize to 0–1
        max_possible = np.log1p(self.config.strength_decay_cap_days) * len(cluster_df)
        return min(raw_strength / max_possible, 1.0)

    def _infer_zone_type(self, cluster_df: pd.DataFrame) -> str:"""
"""
        Decide SUPPORT / RESISTANCE / FLIP.
        """
"""        highs = (cluster_df["type"] == "HIGH").sum()
        lows = (cluster_df["type"] == "LOW").sum()

        if highs > lows:
            return "RESISTANCE"
        elif lows > highs:
            return "SUPPORT"
        else:
            return "FLIP"

    def _label_strength(self, score: float) -> str:
        if score >= 0.8:
            return "VERY_STRONG"
        elif score >= 0.6:
            return "STRONG"
        elif score >= 0.4:
            return "MODERATE"
        else:
            return "WEAK"
"""