"""
HMM Feature Engineering for Funding Rate Regime Detection

Features:
1. Change in basis (perp - spot price change)
2. Change in annualized funding (percentage change)
3. Short MA - Long MA of funding (8h - 24h)
4. Realized volatility (24h rolling)
5. Perp NOFI (Normalized Order Flow Imbalance)
6. Flow divergence (NOFI perp - NOFI spot)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_data(
    spot_path: str = "data/Price/btc-usdt-binance.csv",
    perp_path: str = "data/Price/btc-usdt-perp-binance.csv",
    funding_path: str = "data/Funding Rate/btc-usd-perp_binance.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load spot, perp, and funding rate data."""
    spot = pd.read_csv(spot_path, parse_dates=["TIMESTAMP"])
    perp = pd.read_csv(perp_path, parse_dates=["TIMESTAMP"])
    funding = pd.read_csv(funding_path, parse_dates=["TIMESTAMP"])
    
    spot = spot.sort_values("TIMESTAMP").reset_index(drop=True)
    perp = perp.sort_values("TIMESTAMP").reset_index(drop=True)
    funding = funding.sort_values("TIMESTAMP").reset_index(drop=True)
    
    return spot, perp, funding


def compute_nofi(df: pd.DataFrame) -> pd.Series:
    """
    Compute Normalized Order Flow Imbalance.
    NOFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    """
    buy_vol = df["VOLUME_BUY"] if "VOLUME_BUY" in df.columns else df["TOTAL_TRADES_BUY"]
    sell_vol = df["VOLUME_SELL"] if "VOLUME_SELL" in df.columns else df["TOTAL_TRADES_SELL"]
    
    total_vol = buy_vol + sell_vol
    nofi = (buy_vol - sell_vol) / total_vol
    nofi = nofi.replace([np.inf, -np.inf], np.nan)
    
    return nofi


def compute_basis(spot_close: pd.Series, perp_close: pd.Series) -> pd.Series:
    """Compute basis as perp - spot price."""
    return perp_close - spot_close


def compute_annualized_funding(funding_rate: pd.Series, periods_per_year: int = 1095) -> pd.Series:
    """
    Annualize funding rate.
    Binance funding is paid every 8 hours = 3 times per day = 1095 times per year.
    """
    return funding_rate * periods_per_year * 100


def build_features(
    spot: pd.DataFrame,
    perp: pd.DataFrame,
    funding: pd.DataFrame,
    short_ma_window: int = 8,
    long_ma_window: int = 24,
    rv_window: int = 24
) -> pd.DataFrame:
    """
    Build feature matrix for HMM.
    
    Parameters
    ----------
    spot : pd.DataFrame
        Spot price data with TIMESTAMP, CLOSE, VOLUME_BUY, VOLUME_SELL
    perp : pd.DataFrame
        Perp price data with TIMESTAMP, CLOSE, VOLUME_BUY, VOLUME_SELL
    funding : pd.DataFrame
        Funding rate data with TIMESTAMP, CLOSE (funding rate)
    short_ma_window : int
        Short moving average window for funding (default 8 hours)
    long_ma_window : int
        Long moving average window for funding (default 24 hours)
    rv_window : int
        Realized volatility window (default 24 hours)
    
    Returns
    -------
    pd.DataFrame
        Feature matrix aligned by timestamp
    """
    spot = spot.copy()
    perp = perp.copy()
    funding = funding.copy()
    
    spot = spot.rename(columns={"CLOSE": "spot_close"})
    perp = perp.rename(columns={"CLOSE": "perp_close"})
    funding = funding.rename(columns={"CLOSE": "funding_rate"})
    
    spot["spot_nofi"] = compute_nofi(spot)
    perp["perp_nofi"] = compute_nofi(perp)
    
    spot_cols = ["TIMESTAMP", "spot_close", "spot_nofi", "VOLUME_BUY", "VOLUME_SELL"]
    spot_cols = [c for c in spot_cols if c in spot.columns]
    perp_cols = ["TIMESTAMP", "perp_close", "perp_nofi", "VOLUME_BUY", "VOLUME_SELL"]
    perp_cols = [c for c in perp_cols if c in perp.columns]
    
    df = pd.merge(
        spot[spot_cols],
        perp[perp_cols],
        on="TIMESTAMP",
        how="inner",
        suffixes=("_spot", "_perp")
    )
    
    df = pd.merge(
        df,
        funding[["TIMESTAMP", "funding_rate"]],
        on="TIMESTAMP",
        how="left"
    )
    
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    df["funding_rate"] = df["funding_rate"].ffill()
    
    df["basis"] = compute_basis(df["spot_close"], df["perp_close"])
    df["basis_change"] = df["basis"].diff()
    
    df["annualized_funding"] = compute_annualized_funding(df["funding_rate"])
    df["funding_pct_change"] = df["annualized_funding"].pct_change()
    df["funding_pct_change"] = df["funding_pct_change"].replace([np.inf, -np.inf], np.nan)
    
    df["funding_ma_short"] = df["annualized_funding"].rolling(window=short_ma_window).mean()
    df["funding_ma_long"] = df["annualized_funding"].rolling(window=long_ma_window).mean()
    df["funding_ma_diff"] = df["funding_ma_short"] - df["funding_ma_long"]
    
    df["spot_returns"] = np.log(df["spot_close"] / df["spot_close"].shift(1))
    df["realized_vol"] = df["spot_returns"].rolling(window=rv_window).std() * np.sqrt(365 * 24)
    
    df["flow_divergence"] = df["perp_nofi"] - df["spot_nofi"]
    
    feature_cols = [
        "basis_change",
        "funding_pct_change", 
        "funding_ma_diff",
        "realized_vol",
        "perp_nofi",
        "flow_divergence"
    ]
    
    features = df[["TIMESTAMP"] + feature_cols].copy()
    features = features.dropna()
    
    return features, df


def winsorize_series(s: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Winsorize a series to remove extreme outliers."""
    lower = s.quantile(lower_pct)
    upper = s.quantile(upper_pct)
    return s.clip(lower, upper)


def standardize_features(
    features: pd.DataFrame, 
    feature_cols: list,
    winsorize: bool = True,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> Tuple[pd.DataFrame, dict]:
    """
    Standardize features to zero mean and unit variance.
    Optionally winsorize to handle outliers.
    Returns standardized features and scaling parameters.
    """
    features = features.copy()
    scaling_params = {}
    
    for col in feature_cols:
        if winsorize:
            features[col] = winsorize_series(features[col], lower_pct, upper_pct)
        
        mean = features[col].mean()
        std = features[col].std()
        features[col] = (features[col] - mean) / std
        scaling_params[col] = {"mean": mean, "std": std}
    
    return features, scaling_params


def prepare_hmm_input(
    spot_path: str = "data/Price/btc-usdt-binance.csv",
    perp_path: str = "data/Price/btc-usdt-perp-binance.csv",
    funding_path: str = "data/Funding Rate/btc-usd-perp_binance.csv",
    standardize: bool = True
) -> Tuple[np.ndarray, pd.DataFrame, Optional[dict]]:
    """
    Complete pipeline to prepare HMM input.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features) for HMM
    features_df : pd.DataFrame
        DataFrame with features and timestamps
    scaling_params : dict or None
        Scaling parameters if standardize=True
    """
    spot, perp, funding = load_data(spot_path, perp_path, funding_path)
    features, full_df = build_features(spot, perp, funding)
    
    feature_cols = [
        "basis_change",
        "funding_pct_change",
        "funding_ma_diff", 
        "realized_vol",
        "perp_nofi",
        "flow_divergence"
    ]
    
    scaling_params = None
    if standardize:
        features, scaling_params = standardize_features(features, feature_cols)
    
    X = features[feature_cols].values
    
    return X, features, scaling_params


if __name__ == "__main__":
    X, features_df, scaling_params = prepare_hmm_input()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Date range: {features_df['TIMESTAMP'].min()} to {features_df['TIMESTAMP'].max()}")
    print(f"\nFeature statistics (standardized):")
    print(features_df.describe())
