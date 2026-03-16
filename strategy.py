"""
Signal Generation for Basis Trading Strategy

Combines three signals:
- S1: Z-score dislocation (mean-reversion)
- S2: Volatility filter (activity confirmation)
- S3: HMM regime (fade the crowd)

Entry: S1 and S3 agree on direction, S2 confirms activity
Exit: Regime change, z-score crosses zero, or volatility spike
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class StrategyParams:
    """Parameters for the basis trading strategy."""
    zscore_entry_threshold: float = 1.5
    zscore_exit_threshold: float = 0.5
    vol_ratio_active_pct: float = 0.90
    vol_ratio_spike_pct: float = 0.95
    hmm_confidence_threshold: float = 0.6
    base_position_size: float = 1.0


def compute_S1(zscore: np.ndarray, entry_thresh: float = 1.5) -> np.ndarray:
    """
    S1: Z-score dislocation signal.
    
    Parameters
    ----------
    zscore : np.ndarray
        Basis z-score values
    entry_thresh : float
        Z-score threshold for entry (default 1.5)
    
    Returns
    -------
    np.ndarray
        Signal: +1 (short basis / long spot), -1 (long basis / short spot), 0 (no signal)
    """
    signal = np.zeros(len(zscore))
    signal[zscore > entry_thresh] = 1
    signal[zscore < -entry_thresh] = -1
    return signal


def compute_S2(vol_ratio: np.ndarray, active_thresh: float) -> np.ndarray:
    """
    S2: Volatility filter signal.
    
    Parameters
    ----------
    vol_ratio : np.ndarray
        Volatility ratio (short/long realized vol)
    active_thresh : float
        Threshold for considering market "active"
    
    Returns
    -------
    np.ndarray
        Signal: 1 (active market), 0 (inactive)
    """
    return (vol_ratio > active_thresh).astype(int)


def compute_S3(
    regime: np.ndarray,
    crowd_long_prob: np.ndarray,
    crowd_short_prob: np.ndarray,
    confidence_thresh: float = 0.6
) -> np.ndarray:
    """
    S3: HMM regime signal (fade the crowd).
    
    Parameters
    ----------
    regime : np.ndarray
        Regime labels ('crowd_long', 'crowd_short', 'neutral')
    crowd_long_prob : np.ndarray
        Probability of crowd_long regime
    crowd_short_prob : np.ndarray
        Probability of crowd_short regime
    confidence_thresh : float
        Minimum probability to generate signal
    
    Returns
    -------
    np.ndarray
        Signal: +1 (fade short crowd → go long), -1 (fade long crowd → go short), 0 (neutral)
    """
    signal = np.zeros(len(regime))
    crowd_long_mask = (regime == 'crowd_long') & (crowd_long_prob >= confidence_thresh)
    crowd_short_mask = (regime == 'crowd_short') & (crowd_short_prob >= confidence_thresh)
    signal[crowd_long_mask] = -1  # Fade the long crowd → short
    signal[crowd_short_mask] = 1  # Fade the short crowd → long
    return signal


def compute_entry_signal(S1: np.ndarray, S2: np.ndarray, S3: np.ndarray) -> np.ndarray:
    """
    Combined entry signal.
    
    Entry occurs when:
    - S1 != 0 (z-score dislocation)
    - S2 == 1 (active volatility)
    - S3 != 0 (HMM regime signal)
    - sign(S1) == sign(S3) (signals agree on direction)
    
    Parameters
    ----------
    S1 : np.ndarray
        Z-score signal
    S2 : np.ndarray
        Volatility filter
    S3 : np.ndarray
        HMM regime signal
    
    Returns
    -------
    np.ndarray
        Entry signal: +1 (long), -1 (short), 0 (no entry)
    """
    entry = np.zeros(len(S1))
    valid = (S1 != 0) & (S2 == 1) & (S3 != 0) & (np.sign(S1) == np.sign(S3))
    entry[valid] = S3[valid]
    return entry


def compute_exit_signal(
    regime: np.ndarray,
    regime_prev: np.ndarray,
    zscore: np.ndarray,
    vol_ratio: np.ndarray,
    vol_spike_thresh: float
) -> np.ndarray:
    """
    Exit signal based on multiple conditions.
    
    Exit occurs when ANY of:
    - Regime changes
    - Z-score crosses zero
    - Volatility spikes above threshold
    
    Parameters
    ----------
    regime : np.ndarray
        Current regime labels
    regime_prev : np.ndarray
        Previous regime labels
    zscore : np.ndarray
        Basis z-score values
    vol_ratio : np.ndarray
        Volatility ratio
    vol_spike_thresh : float
        Threshold for volatility spike
    
    Returns
    -------
    np.ndarray
        Exit signal: 1 (exit), 0 (hold)
    """
    zscore_prev = np.roll(zscore, 1)
    zscore_prev[0] = zscore[0]
    
    regime_change = regime != regime_prev
    zscore_cross = np.sign(zscore) != np.sign(zscore_prev)
    vol_spike = vol_ratio > vol_spike_thresh
    
    return (regime_change | zscore_cross | vol_spike).astype(int)


def compute_position_size(S2: np.ndarray, base_size: float = 1.0) -> np.ndarray:
    """
    Position sizing based on volatility regime.
    
    Parameters
    ----------
    S2 : np.ndarray
        Volatility filter (1 = active, 0 = inactive)
    base_size : float
        Base position size (default 1.0 = 100%)
    
    Returns
    -------
    np.ndarray
        Position size: full when active, 50% when inactive
    """
    return np.where(S2 == 1, base_size, base_size * 0.5)


def generate_signals(
    df: pd.DataFrame,
    regime: np.ndarray,
    crowd_long_prob: np.ndarray,
    crowd_short_prob: np.ndarray,
    params: Optional[StrategyParams] = None,
    vol_ratio_active_thresh: Optional[float] = None,
    vol_ratio_spike_thresh: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate all trading signals for a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'basis_zscore' and 'vol_ratio' columns
    regime : np.ndarray
        HMM regime labels
    crowd_long_prob : np.ndarray
        Crowd long probabilities
    crowd_short_prob : np.ndarray
        Crowd short probabilities
    params : StrategyParams, optional
        Strategy parameters (uses defaults if None)
    vol_ratio_active_thresh : float, optional
        Override volatility active threshold (computed from data if None)
    vol_ratio_spike_thresh : float, optional
        Override volatility spike threshold (computed from data if None)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with signal columns added
    """
    if params is None:
        params = StrategyParams()
    
    result = df.copy()
    
    # Compute vol thresholds from data if not provided
    if vol_ratio_active_thresh is None:
        vol_ratio_active_thresh = result['vol_ratio'].quantile(params.vol_ratio_active_pct)
    if vol_ratio_spike_thresh is None:
        vol_ratio_spike_thresh = result['vol_ratio'].quantile(params.vol_ratio_spike_pct)
    
    # Generate individual signals
    result['S1'] = compute_S1(
        result['basis_zscore'].fillna(0).values,
        params.zscore_entry_threshold
    )
    result['S2'] = compute_S2(
        result['vol_ratio'].fillna(0).values,
        vol_ratio_active_thresh
    )
    result['S3'] = compute_S3(
        regime,
        crowd_long_prob,
        crowd_short_prob,
        params.hmm_confidence_threshold
    )
    
    # Combined signals
    result['entry_signal'] = compute_entry_signal(
        result['S1'].values,
        result['S2'].values,
        result['S3'].values
    )
    
    # Exit signal (needs previous regime)
    regime_prev = np.roll(regime, 1)
    regime_prev[0] = regime[0]
    result['exit_signal'] = compute_exit_signal(
        regime,
        regime_prev,
        result['basis_zscore'].fillna(0).values,
        result['vol_ratio'].fillna(0).values,
        vol_ratio_spike_thresh
    )
    
    result['position_size'] = compute_position_size(
        result['S2'].values,
        params.base_position_size
    )
    
    # Store regime info
    result['regime'] = regime
    result['crowd_long_prob'] = crowd_long_prob
    result['crowd_short_prob'] = crowd_short_prob
    
    return result


def signals_to_positions(
    entry_signal: np.ndarray,
    exit_signal: np.ndarray,
    position_size: np.ndarray
) -> np.ndarray:
    """
    Convert entry/exit signals to position array.
    
    Parameters
    ----------
    entry_signal : np.ndarray
        Entry signals (+1 long, -1 short, 0 none)
    exit_signal : np.ndarray
        Exit signals (1 exit, 0 hold)
    position_size : np.ndarray
        Position size multiplier
    
    Returns
    -------
    np.ndarray
        Position array: positive = long, negative = short, 0 = flat
    """
    n = len(entry_signal)
    position = np.zeros(n)
    current_pos = 0.0
    
    for i in range(n):
        if exit_signal[i] == 1:
            current_pos = 0.0
        
        if entry_signal[i] != 0:
            current_pos = entry_signal[i] * position_size[i]
        
        position[i] = current_pos
    
    return position

if __name__ == "__main__":
    print("Strategy module loaded.")
    print(f"Default params: {StrategyParams()}")

