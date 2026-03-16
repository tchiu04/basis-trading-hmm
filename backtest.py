"""
VectorBT Walk-Forward Backtesting for Basis Trading Strategy

Provides clean walk-forward validation with:
- Rolling window train/test splits
- HMM regime integration
- VectorBT portfolio metrics
- Out-of-sample performance tracking
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import warnings

from hmm import FundingRegimeHMM
from hmm_features import load_data, build_features
from strategy import (
    StrategyParams,
    compute_S1,
    compute_S2,
    compute_S3,
    compute_entry_signal,
    compute_exit_signal,
    compute_position_size,
    signals_to_positions
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SimplePortfolio:
    """
    Simple portfolio wrapper for when VectorBT API is incompatible.
    Provides basic stats from a returns series.
    """
    
    def __init__(self, returns: pd.Series, init_cash: float = 100_000):
        self.returns = returns.fillna(0)
        self.init_cash = init_cash
        self._cumulative = (1 + self.returns).cumprod()
        self._value = self.init_cash * self._cumulative
    
    def value(self) -> pd.Series:
        """Portfolio value over time."""
        return self._value
    
    def total_return(self) -> float:
        """Total return as decimal."""
        return self._cumulative.iloc[-1] - 1 if len(self._cumulative) > 0 else 0
    
    def drawdown(self) -> pd.Series:
        """Drawdown series."""
        rolling_max = self._value.cummax()
        dd = (self._value - rolling_max) / rolling_max
        return dd.fillna(0)
    
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        return self.drawdown().min()
    
    def sharpe_ratio(self, periods_per_year: int = 365 * 24) -> float:
        """Annualized Sharpe ratio."""
        if self.returns.std() == 0:
            return 0.0
        return self.returns.mean() / self.returns.std() * np.sqrt(periods_per_year)
    
    def stats(self) -> pd.Series:
        """Basic portfolio statistics."""
        total_ret = self.total_return()
        max_dd = self.max_drawdown()
        sharpe = self.sharpe_ratio()
        n_nonzero = (self.returns != 0).sum()
        
        return pd.Series({
            'Start': self.returns.index[0] if len(self.returns) > 0 else None,
            'End': self.returns.index[-1] if len(self.returns) > 0 else None,
            'Total Return [%]': total_ret * 100,
            'Max Drawdown [%]': max_dd * 100,
            'Sharpe Ratio': sharpe,
            'Win Rate [%]': (self.returns > 0).sum() / max(n_nonzero, 1) * 100,
            'Best Return [%]': self.returns.max() * 100,
            'Worst Return [%]': self.returns.min() * 100,
            'Avg Return [%]': self.returns.mean() * 100,
        })


@dataclass
class CostParams:
    """Trading cost parameters."""
    spot_fee: float = 0.001       # Spot trading fee (0.1% = taker fee)
    perp_fee: float = 0.0005      # Perp trading fee (0.05% = maker rebate typical)
    spot_slippage: float = 0.0005 # Spot slippage estimate
    perp_slippage: float = 0.0002 # Perp slippage estimate
    borrow_rate_daily: float = 0.0 # Daily borrow rate for short spot (Stage 2)
    
    @property
    def spot_cost(self) -> float:
        """Total spot transaction cost rate."""
        return self.spot_fee + self.spot_slippage
    
    @property
    def perp_cost(self) -> float:
        """Total perp transaction cost rate."""
        return self.perp_fee + self.perp_slippage


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    train_months: int = 8
    test_months: int = 1
    n_states: int = 3
    strategy_params: StrategyParams = field(default_factory=StrategyParams)
    cost_params: CostParams = field(default_factory=CostParams)
    quantity: float = 1.0         # Quantity per leg in BTC (default 1 BTC)
    freq: str = '1h'
    point_in_time_hmm: bool = True
    funding_interval_hours: int = 8  # Binance funding every 8 hours


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_samples: int
    n_entries: int
    df: pd.DataFrame
    portfolio: Optional[Any] = None


@dataclass
class WalkForwardResult:
    """Combined results from walk-forward backtest."""
    folds: List[FoldResult]
    df_combined: pd.DataFrame
    portfolio: Optional[Any] = None
    
    @property
    def n_folds(self) -> int:
        return len(self.folds)
    
    @property
    def total_samples(self) -> int:
        return len(self.df_combined)
    
    @property
    def total_entries(self) -> int:
        return (self.df_combined['entry_signal'] != 0).sum()
    
    @property
    def total_pnl(self) -> float:
        """Total PnL across all folds."""
        if 'total_pnl' in self.df_combined.columns:
            return self.df_combined['total_pnl'].sum()
        return 0.0
    
    @property
    def total_basis_pnl(self) -> float:
        """Total basis convergence PnL."""
        if 'basis_pnl' in self.df_combined.columns:
            return self.df_combined['basis_pnl'].sum()
        return 0.0
    
    @property
    def total_funding_pnl(self) -> float:
        """Total funding PnL."""
        if 'funding_pnl' in self.df_combined.columns:
            return self.df_combined['funding_pnl'].sum()
        return 0.0
    
    def summary(self) -> pd.DataFrame:
        """Generate summary statistics per fold."""
        records = []
        for f in self.folds:
            records.append({
                'fold': f.fold,
                'train_start': f.train_start,
                'train_end': f.train_end,
                'test_start': f.test_start,
                'test_end': f.test_end,
                'n_samples': f.n_samples,
                'n_entries': f.n_entries
            })
        return pd.DataFrame(records)
    
    def pnl_summary(self) -> Dict[str, Any]:
        """Generate PnL summary statistics."""
        df = self.df_combined
        
        if 'total_pnl' not in df.columns:
            return {'error': 'PnL not computed'}
        
        # --- PnL breakdown ---
        total_pnl = df['total_pnl'].sum()
        spot_pnl = df['spot_pnl'].sum() if 'spot_pnl' in df.columns else 0
        perp_pnl = df['perp_pnl'].sum() if 'perp_pnl' in df.columns else 0
        m2m_pnl = df['m2m_pnl'].sum() if 'm2m_pnl' in df.columns else (spot_pnl + perp_pnl)
        funding_pnl = df['funding_pnl'].sum() if 'funding_pnl' in df.columns else 0
        total_cost = df['total_cost'].sum() if 'total_cost' in df.columns else 0
        
        # --- Returns stats ---
        returns = df['strategy_returns'] if 'strategy_returns' in df.columns else df['total_pnl']
        returns = returns.dropna()
        
        # Sharpe (annualized, assuming hourly data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24)
        else:
            sharpe = 0.0
        
        # Total return
        total_return = returns.sum()
        
        # --- Drawdown ---
        cumulative = df['cumulative_pnl'] if 'cumulative_pnl' in df.columns else df['total_pnl'].cumsum()
        rolling_max = cumulative.cummax()
        drawdown = cumulative - rolling_max
        max_dd = drawdown.min()
        max_dd_pct = max_dd / rolling_max.max() if rolling_max.max() > 0 else 0
        
        # --- Trade statistics ---
        # Count position changes as trades
        if 'direction' in df.columns:
            direction_changes = df['direction'].diff().fillna(0) != 0
        else:
            direction_changes = df['position'].diff().fillna(0) != 0
        
        n_trades = direction_changes.sum()
        
        # Time in position
        in_position = (df.get('direction', df.get('position', pd.Series([0]))) != 0)
        pct_time_in_position = in_position.mean()
        
        # Win rate per bar (when in position)
        if in_position.sum() > 0:
            win_rate = (df.loc[in_position, 'total_pnl'] > 0).mean()
        else:
            win_rate = 0
        
        return {
            'total_pnl': total_pnl,
            'spot_pnl': spot_pnl,
            'perp_pnl': perp_pnl,
            'm2m_pnl': m2m_pnl,
            'funding_pnl': funding_pnl,
            'total_cost': total_cost,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'n_trades': int(n_trades),
            'n_entries': self.total_entries,
            'pct_time_in_position': pct_time_in_position,
            'win_rate': win_rate,
            'avg_pnl_per_hour': total_pnl / len(df) if len(df) > 0 else 0
        }


class WalkForwardBacktest:
    """
    Walk-forward backtester with HMM regime integration.
    
    Uses VectorBT for portfolio simulation and metrics.
    """
    
    HMM_FEATURE_COLS = [
        'basis_change', 'funding_pct_change', 'funding_ma_diff',
        'realized_vol', 'perp_nofi', 'flow_divergence'
    ]
    
    def __init__(
        self,
        df: pd.DataFrame,
        hmm_features: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None
    ):
        """
        Initialize walk-forward backtester.
        
        Parameters
        ----------
        df : pd.DataFrame
            Main strategy DataFrame with TIMESTAMP, basis_zscore, vol_ratio, CLOSE columns
        hmm_features : pd.DataFrame
            HMM feature DataFrame with TIMESTAMP and feature columns
        config : WalkForwardConfig, optional
            Walk-forward configuration
        """
        self.df = df.copy()
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'])
        
        self.hmm_features = hmm_features.copy()
        self.hmm_features['TIMESTAMP'] = pd.to_datetime(self.hmm_features['TIMESTAMP'])
        
        self.config = config or WalkForwardConfig()
        self.results: Optional[WalkForwardResult] = None
    
    def run(
        self,
        train_start: Optional[pd.Timestamp] = None,
        test_end: Optional[pd.Timestamp] = None,
        verbose: bool = True
    ) -> WalkForwardResult:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        train_start : pd.Timestamp, optional
            Start of first training window (defaults to data start + train_months)
        test_end : pd.Timestamp, optional
            End of testing (defaults to data end)
        verbose : bool
            Print progress
        
        Returns
        -------
        WalkForwardResult
            Combined results from all folds
        """
        timestamps = self.df['TIMESTAMP']
        
        if train_start is None:
            train_start = timestamps.min()
        
        train_end = train_start + pd.DateOffset(months=self.config.train_months)
        
        if test_end is None:
            test_end = timestamps.max()
        
        all_folds: List[FoldResult] = []
        fold = 0
        
        while train_end < test_end:
            fold_test_end = min(
                train_end + pd.DateOffset(months=self.config.test_months),
                test_end + pd.Timedelta(days=1)
            )
            
            fold_result = self._run_single_fold(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                test_end=fold_test_end
            )
            
            if fold_result is None:
                break
            
            all_folds.append(fold_result)
            
            if verbose:
                print(
                    f"Fold {fold}: Train {train_start.date()} to {train_end.date()}, "
                    f"Test {train_end.date()} to {fold_test_end.date()} "
                    f"({fold_result.n_samples} samples, {fold_result.n_entries} entries)"
                )
            
            # Roll forward
            train_start = train_start + pd.DateOffset(months=self.config.test_months)
            train_end = fold_test_end
            fold += 1
        
        # Combine all folds
        df_combined = pd.concat([f.df for f in all_folds], ignore_index=True)
        
        # Compute basis trade PnL
        df_combined = self._compute_basis_pnl(df_combined)
        
        # Create combined portfolio
        portfolio = self._create_portfolio(df_combined)
        
        self.results = WalkForwardResult(
            folds=all_folds,
            df_combined=df_combined,
            portfolio=portfolio
        )
        
        if verbose:
            print(f"\nTotal walk-forward samples: {self.results.total_samples}")
            print(f"Total entry signals: {self.results.total_entries}")
        
        return self.results
    
    def _run_single_fold(
        self,
        fold: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_end: pd.Timestamp
    ) -> Optional[FoldResult]:
        """Run a single walk-forward fold."""
        
        # Create masks
        train_mask = (self.df['TIMESTAMP'] >= train_start) & (self.df['TIMESTAMP'] < train_end)
        test_mask = (self.df['TIMESTAMP'] >= train_end) & (self.df['TIMESTAMP'] < test_end)
        
        hmm_train_mask = (
            (self.hmm_features['TIMESTAMP'] >= train_start) & 
            (self.hmm_features['TIMESTAMP'] < train_end)
        )
        hmm_test_mask = (
            (self.hmm_features['TIMESTAMP'] >= train_end) & 
            (self.hmm_features['TIMESTAMP'] < test_end)
        )
        
        if test_mask.sum() == 0 or hmm_test_mask.sum() == 0:
            return None
        
        # Get training data for threshold calculation
        df_train = self.df[train_mask]
        vol_ratio_active_thresh = df_train['vol_ratio'].quantile(
            self.config.strategy_params.vol_ratio_active_pct
        )
        vol_ratio_spike_thresh = df_train['vol_ratio'].quantile(
            self.config.strategy_params.vol_ratio_spike_pct
        )
        
        # HMM: Train and predict
        X_hmm_train = self.hmm_features[hmm_train_mask][self.HMM_FEATURE_COLS].values
        X_hmm_test = self.hmm_features[hmm_test_mask][self.HMM_FEATURE_COLS].values
        
        # Standardize using training statistics
        hmm_mean = X_hmm_train.mean(axis=0)
        hmm_std = X_hmm_train.std(axis=0)
        X_hmm_train_scaled = (X_hmm_train - hmm_mean) / hmm_std
        X_hmm_test_scaled = (X_hmm_test - hmm_mean) / hmm_std
        
        # Fit HMM
        regime_labels, crowd_long_prob, crowd_short_prob, neutral_prob = self._fit_hmm_fold(
            X_hmm_train_scaled, X_hmm_test_scaled
        )
        
        # Create HMM results DataFrame
        hmm_fold_df = pd.DataFrame({
            'TIMESTAMP': self.hmm_features[hmm_test_mask]['TIMESTAMP'].values,
            'regime': regime_labels,
            'crowd_long_prob': crowd_long_prob,
            'crowd_short_prob': crowd_short_prob,
            'neutral_prob': neutral_prob
        })
        hmm_fold_df['TIMESTAMP'] = pd.to_datetime(hmm_fold_df['TIMESTAMP'])
        
        # Get test data
        df_test = self.df[test_mask].copy()
        df_test['TIMESTAMP'] = pd.to_datetime(df_test['TIMESTAMP'])
        
        # Drop any existing HMM columns
        hmm_cols = ['regime', 'crowd_long_prob', 'crowd_short_prob', 'neutral_prob',
                    'S1', 'S2', 'S3', 'entry_signal', 'exit_signal', 'position_size']
        df_test = df_test.drop(columns=[c for c in hmm_cols if c in df_test.columns], errors='ignore')
        
        # Merge HMM results
        df_test = df_test.merge(hmm_fold_df, on='TIMESTAMP', how='left')
        
        # Fill missing values
        df_test['regime'] = df_test['regime'].fillna('neutral')
        df_test['crowd_long_prob'] = df_test['crowd_long_prob'].fillna(0.0)
        df_test['crowd_short_prob'] = df_test['crowd_short_prob'].fillna(0.0)
        df_test['neutral_prob'] = df_test['neutral_prob'].fillna(1.0)
        
        # Generate signals
        df_test['S1'] = compute_S1(
            df_test['basis_zscore'].fillna(0).values,
            self.config.strategy_params.zscore_entry_threshold
        )
        df_test['S2'] = compute_S2(
            df_test['vol_ratio'].fillna(0).values,
            vol_ratio_active_thresh
        )
        df_test['S3'] = compute_S3(
            df_test['regime'].values,
            df_test['crowd_long_prob'].values,
            df_test['crowd_short_prob'].values,
            self.config.strategy_params.hmm_confidence_threshold
        )
        df_test['entry_signal'] = compute_entry_signal(
            df_test['S1'].values,
            df_test['S2'].values,
            df_test['S3'].values
        )
        df_test['position_size'] = compute_position_size(
            df_test['S2'].values,
            self.config.strategy_params.base_position_size
        )
        
        # Add fold metadata
        df_test['fold'] = fold
        df_test['train_start'] = train_start
        df_test['train_end'] = train_end
        df_test['vol_thresh_active'] = vol_ratio_active_thresh
        df_test['vol_thresh_spike'] = vol_ratio_spike_thresh
        
        return FoldResult(
            fold=fold,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            n_samples=len(df_test),
            n_entries=(df_test['entry_signal'] != 0).sum(),
            df=df_test
        )
    
    def _fit_hmm_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit HMM on training data and predict on test data."""
        
        model = FundingRegimeHMM(n_states=self.config.n_states)
        model.fit(X_train)
        
        n_test = len(X_test)
        n_states = self.config.n_states
        
        if self.config.point_in_time_hmm:
            # Point-in-time: no look-ahead within test window
            states = np.zeros(n_test, dtype=int)
            posteriors = np.zeros((n_test, n_states))
            
            for t in range(n_test):
                X_up_to_t = X_test[:t+1]
                post_t = model.predict_proba(X_up_to_t)
                posteriors[t] = post_t[-1]
                states[t] = np.argmax(posteriors[t])
        else:
            # Batch prediction
            states = model.predict(X_test)
            posteriors = model.predict_proba(X_test)
        
        regime_labels = model.get_regime_labels(states)
        regime_probs = model.get_regime_probabilities(posteriors)
        
        crowd_long_prob = regime_probs['crowd_long'].values
        crowd_short_prob = regime_probs['crowd_short'].values
        neutral_prob = regime_probs.get('neutral', pd.Series(np.zeros(n_test))).values
        
        return regime_labels, crowd_long_prob, crowd_short_prob, neutral_prob
    
    def _create_portfolio(self, df: pd.DataFrame) -> Any:
        """Create VectorBT portfolio from basis trade PnL."""
        
        if 'strategy_returns' not in df.columns:
            return None
        
        # Create portfolio from returns
        df_indexed = df.set_index('TIMESTAMP')
        returns = df_indexed['strategy_returns']
        
        try:
            # Try VectorBT Portfolio.from_returns if available
            if hasattr(vbt.Portfolio, 'from_returns'):
                portfolio = vbt.Portfolio.from_returns(
                    returns=returns,
                    freq=self.config.freq,
                    init_cash=100_000
                )
            else:
                # Use SimplePortfolio fallback
                portfolio = SimplePortfolio(returns, init_cash=100_000)
        except Exception as e:
            print(f"Warning: VectorBT portfolio creation failed: {e}")
            portfolio = SimplePortfolio(returns, init_cash=100_000)
        
        return portfolio
    
    def _compute_basis_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute basis trade PnL with proper leg-by-leg accounting.
        
        Position convention:
        - d_t = +1: "fade crowded long" → long spot, short perp
        - d_t = -1: "fade crowded short" → short spot, long perp
        - d_t = 0: flat
        
        PnL components:
        1. Spot PnL: q_spot * (P_spot[t+1] - P_spot[t])
        2. Perp PnL: q_perp * (P_perp[t+1] - P_perp[t])
        3. Funding PnL: -f_t * Notional_perp (only at funding times)
        4. Costs: spot + perp transaction costs
        
        Stage 1 implementation (clean minimal version).
        """
        result = df.copy()
        
        # Get price columns
        spot_col = 'spot_close' if 'spot_close' in result.columns else 'CLOSE_spot'
        perp_col = 'perp_close' if 'perp_close' in result.columns else 'CLOSE_perp'
        
        # Check for required columns
        has_spot = spot_col in result.columns
        has_perp = perp_col in result.columns
        
        if not has_spot or not has_perp:
            # Fallback: try to get from CLOSE_coin (spot proxy)
            if 'CLOSE_coin' in result.columns and 'basis' in result.columns:
                result['spot_close'] = result['CLOSE_coin']
                result['perp_close'] = result['CLOSE_coin'] + result['basis']
                spot_col = 'spot_close'
                perp_col = 'perp_close'
            else:
                return result
        
        Q = self.config.quantity  # Quantity per leg (default 1 BTC)
        cost_params = self.config.cost_params
        
        # --- 1. Direction signal d_t ∈ {-1, 0, +1} ---
        # Convert entry signals to positions (with carry-forward)
        d = signals_to_positions(
            result['entry_signal'].values,
            np.zeros(len(result)),
            np.ones(len(result))  # Use unit position, scale by quantity
        )
        result['direction'] = d
        
        # --- 2. Compute leg quantities ---
        # Fixed quantity Q (e.g., 1 BTC) per leg
        # q_spot = d * Q  (positive when long spot)
        # q_perp = -d * Q (negative when short perp, i.e., opposite of spot)
        P_spot = result[spot_col].values
        P_perp = result[perp_col].values
        
        q_spot = d * Q
        q_perp = -d * Q
        
        result['q_spot'] = q_spot
        result['q_perp'] = q_perp
        
        # Notional values (for reference)
        result['notional_spot'] = q_spot * P_spot  # = d * Q * P_spot
        result['notional_perp'] = q_perp * P_perp  # = -d * Q * P_perp
        
        # --- 3. Price changes ---
        dP_spot = np.diff(P_spot, prepend=P_spot[0])
        dP_perp = np.diff(P_perp, prepend=P_perp[0])
        
        # --- 4. Spot PnL: q_spot[t-1] * (P_spot[t] - P_spot[t-1]) ---
        q_spot_prev = np.roll(q_spot, 1)
        q_spot_prev[0] = 0
        result['spot_pnl'] = q_spot_prev * dP_spot
        
        # --- 5. Perp PnL: q_perp[t-1] * (P_perp[t] - P_perp[t-1]) ---
        q_perp_prev = np.roll(q_perp, 1)
        q_perp_prev[0] = 0
        result['perp_pnl'] = q_perp_prev * dP_perp
        
        # --- 6. Mark-to-market PnL (spot + perp) ---
        result['m2m_pnl'] = result['spot_pnl'] + result['perp_pnl']
        
        # --- 7. Funding PnL ---
        # PnL_funding = -f_t * Notional_perp
        # Notional_perp = q_perp * P_perp
        # Only apply at funding timestamps (every 8 hours for Binance)
        result['funding_pnl'] = 0.0
        
        if 'funding_rate' in result.columns:
            # Determine funding timestamps (00:00, 08:00, 16:00 UTC)
            if 'TIMESTAMP' in result.columns:
                hours = pd.to_datetime(result['TIMESTAMP']).dt.hour
                is_funding_time = hours.isin([0, 8, 16])
            else:
                # Fallback: assume every 8th bar if hourly
                is_funding_time = np.arange(len(result)) % self.config.funding_interval_hours == 0
            
            # Perp notional at previous bar: q_perp * P_perp
            q_perp_prev = np.roll(q_perp, 1)
            q_perp_prev[0] = 0
            notional_perp_prev = q_perp_prev * P_perp
            
            # Funding cashflow: -f * Notional_perp
            # If short perp (q_perp < 0) and f > 0: receive funding (positive PnL)
            # If long perp (q_perp > 0) and f > 0: pay funding (negative PnL)
            funding_pnl = -result['funding_rate'].values * notional_perp_prev
            funding_pnl = np.where(is_funding_time, funding_pnl, 0)
            result['funding_pnl'] = funding_pnl
        
        # --- 8. Trading costs ---
        # Cost = c * |Δq| * P
        dq_spot = np.diff(q_spot, prepend=0)
        dq_perp = np.diff(q_perp, prepend=0)
        
        result['cost_spot'] = cost_params.spot_cost * np.abs(dq_spot) * P_spot
        result['cost_perp'] = cost_params.perp_cost * np.abs(dq_perp) * P_perp
        result['total_cost'] = result['cost_spot'] + result['cost_perp']
        
        # --- 9. Total PnL per bar ---
        result['total_pnl'] = (
            result['spot_pnl'] + 
            result['perp_pnl'] + 
            result['funding_pnl'] - 
            result['total_cost']
        )
        
        # --- 10. Cumulative PnL ---
        result['cumulative_pnl'] = result['total_pnl'].cumsum()
        
        # --- 11. Returns (as fraction of total notional = 2 * Q * avg_price) ---
        avg_price = (P_spot.mean() + P_perp.mean()) / 2
        total_notional = 2 * Q * avg_price  # Approximate total capital deployed
        result['strategy_returns'] = result['total_pnl'] / total_notional
        
        # --- Legacy compatibility ---
        result['position'] = d  # Alias for direction
        result['basis_pnl'] = result['m2m_pnl']  # Alias for m2m
        
        return result


def load_strategy_data(
    spot_path: str = "data/Price/btc-usdt-binance.csv",
    perp_path: str = "data/Price/btc-usdt-perp-binance.csv",
    funding_path: str = "data/Funding Rate/btc-usd-perp_binance.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for backtesting.
    
    Returns
    -------
    df : pd.DataFrame
        Main strategy DataFrame with features
    hmm_features : pd.DataFrame
        HMM feature DataFrame
    """
    spot, perp, funding = load_data(spot_path, perp_path, funding_path)
    hmm_features, _ = build_features(spot, perp, funding)
    
    # Build main df (simplified - you may want to use your notebook's feature engineering)
    df = pd.merge(
        spot[['TIMESTAMP', 'CLOSE']].rename(columns={'CLOSE': 'spot_close'}),
        perp[['TIMESTAMP', 'CLOSE']].rename(columns={'CLOSE': 'perp_close'}),
        on='TIMESTAMP',
        how='inner'
    )
    
    # Basic features for demonstration
    df['basis'] = df['perp_close'] - df['spot_close']
    df['basis_zscore'] = (df['basis'] - df['basis'].rolling(168).mean()) / df['basis'].rolling(168).std()
    df['log_returns'] = np.log(df['spot_close'] / df['spot_close'].shift(1))
    df['realized_vol_24h'] = df['log_returns'].rolling(24).std() * np.sqrt(365 * 24)
    df['realized_vol_168h'] = df['log_returns'].rolling(168).std() * np.sqrt(365 * 24)
    df['vol_ratio'] = df['realized_vol_24h'] / df['realized_vol_168h']
    df['CLOSE'] = df['spot_close']
    
    df = df.dropna()
    
    return df, hmm_features


def run_walk_forward(
    df: pd.DataFrame,
    hmm_features: pd.DataFrame,
    config: Optional[WalkForwardConfig] = None,
    train_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
    verbose: bool = True
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward backtest.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main strategy DataFrame
    hmm_features : pd.DataFrame
        HMM features DataFrame
    config : WalkForwardConfig, optional
        Configuration
    train_start : pd.Timestamp, optional
        Start of first training window
    test_end : pd.Timestamp, optional
        End of testing
    verbose : bool
        Print progress
    
    Returns
    -------
    WalkForwardResult
        Walk-forward results
    """
    backtester = WalkForwardBacktest(df, hmm_features, config)
    return backtester.run(train_start, test_end, verbose)


def print_portfolio_stats(result: WalkForwardResult) -> None:
    """Print portfolio statistics."""
    if result.portfolio is None:
        print("No portfolio available.")
        return
    
    print("\n" + "=" * 60)
    print("WALK-FORWARD PORTFOLIO STATISTICS")
    print("=" * 60)
    print(result.portfolio.stats())


if __name__ == "__main__":
    print("Loading data...")
    df, hmm_features = load_strategy_data()
    
    print(f"Data shape: {df.shape}")
    print(f"HMM features shape: {hmm_features.shape}")
    
    config = WalkForwardConfig(
        train_months=8,
        test_months=1,
        strategy_params=StrategyParams(
            zscore_entry_threshold=2.0,
            hmm_confidence_threshold=0.6
        )
    )
    
    print("\nRunning walk-forward backtest...")
    print("-" * 60)
    
    result = run_walk_forward(df, hmm_features, config)
    
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    print(result.summary().to_string(index=False))
    
    print_portfolio_stats(result)
    
    # Save results
    result.df_combined.to_csv("wf_results.csv", index=False)
    print("\nResults saved to wf_results.csv")
