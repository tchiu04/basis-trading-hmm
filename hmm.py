"""
HMM Regime Detection Model for Funding Rate Analysis

Uses hmmlearn's GaussianHMM for regime detection with 3 states:
- Crowd Long: High funding, positive basis, bullish flow
- Crowd Short: Negative funding, negative basis, bearish flow  
- Neutral: Low absolute funding, balanced flow

Outputs:
- Regime labels
- Regime probabilities
- Trading signals
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Dict, List
import warnings
import pickle

from hmm_features import prepare_hmm_input, load_data, build_features, standardize_features

warnings.filterwarnings("ignore", category=DeprecationWarning)


class FundingRegimeHMM:
    """
    Hidden Markov Model for funding rate regime detection.
    
    States are labeled post-hoc based on emission characteristics:
    - Crowd Long: typically high positive funding
    - Crowd Short: typically negative funding
    - Neutral: low absolute funding
    """
    
    REGIME_NAMES = {
        0: "crowd_long",
        1: "crowd_short", 
        2: "neutral"
    }
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        tol: float = 1e-4,
        init_params: str = "stmc",
        params: str = "stmc"
    ):
        """
        Initialize the HMM model.
        
        Parameters
        ----------
        n_states : int
            Number of hidden states (default 3)
        covariance_type : str
            Type of covariance matrix ('full', 'diag', 'spherical', 'tied')
        n_iter : int
            Maximum number of EM iterations
        random_state : int
            Random seed for reproducibility
        tol : float
            Convergence threshold for EM
        init_params : str
            Parameters to initialize ('s'=startprob, 't'=transmat, 'm'=means, 'c'=covars)
        params : str
            Parameters to update during training
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
            init_params=init_params,
            params=params
        )
        self.fitted = False
        self.state_mapping = {}
        self.scaling_params = None
        self.feature_cols = [
            "basis_change",
            "funding_pct_change",
            "funding_ma_diff",
            "realized_vol", 
            "perp_nofi",
            "flow_divergence"
        ]
    
    def fit(
        self, 
        X: np.ndarray, 
        lengths: Optional[list] = None,
        kmeans_init: bool = True
    ) -> "FundingRegimeHMM":
        """
        Fit the HMM model to the feature matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        lengths : list, optional
            Lengths of individual sequences if training on multiple sequences
        kmeans_init : bool
            Use K-means to initialize emission means (helps find better regimes)
        
        Returns
        -------
        self
        """
        if kmeans_init:
            self._kmeans_initialize(X)
        
        self.model.fit(X, lengths)
        self.fitted = True
        self._label_states(X)
        return self
    
    def _kmeans_initialize(self, X: np.ndarray) -> None:
        """Initialize HMM means using K-means clustering."""
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        
        self.model.means_ = kmeans.cluster_centers_
        self.model.init_params = self.model.init_params.replace("m", "")
    
    def _label_states(self, X: np.ndarray) -> None:
        """
        Label states based on their emission characteristics.
        Uses the funding_ma_diff feature (index 2) to determine crowd positioning.
        
        For n_states=3: crowd_long, neutral, crowd_short
        For n_states>3: crowd_long, neutral_high, neutral_mid, ..., crowd_short
        """
        means = self.model.means_
        funding_ma_diff_idx = 2
        
        funding_means = means[:, funding_ma_diff_idx]
        sorted_indices = np.argsort(funding_means)
        
        n = self.n_states
        
        if n == 3:
            self.state_mapping = {
                sorted_indices[2]: "crowd_long",
                sorted_indices[0]: "crowd_short",
                sorted_indices[1]: "neutral"
            }
        elif n == 2:
            self.state_mapping = {
                sorted_indices[1]: "crowd_long",
                sorted_indices[0]: "crowd_short"
            }
        else:
            self.state_mapping = {sorted_indices[-1]: "crowd_long"}
            self.state_mapping[sorted_indices[0]] = "crowd_short"
            
            for i, idx in enumerate(sorted_indices[1:-1]):
                self.state_mapping[idx] = f"neutral_{i+1}"
        
        self.regime_to_state = {v: k for k, v in self.state_mapping.items()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely state sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns
        -------
        states : np.ndarray
            Predicted state sequence (raw state indices)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probability of each state for each observation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns
        -------
        posteriors : np.ndarray
            Posterior probabilities (n_samples, n_states)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_regime_labels(self, states: np.ndarray) -> np.ndarray:
        """
        Convert raw state indices to regime labels.
        
        Parameters
        ----------
        states : np.ndarray
            Raw state indices from predict()
        
        Returns
        -------
        labels : np.ndarray
            Regime labels ('crowd_long', 'crowd_short', 'neutral')
        """
        return np.array([self.state_mapping[s] for s in states])
    
    def get_regime_probabilities(self, posteriors: np.ndarray) -> pd.DataFrame:
        """
        Convert posteriors to labeled regime probabilities.
        
        Parameters
        ----------
        posteriors : np.ndarray
            Posterior probabilities from predict_proba()
        
        Returns
        -------
        probs_df : pd.DataFrame
            DataFrame with columns for each regime probability
        """
        probs_df = pd.DataFrame(posteriors)
        probs_df.columns = [self.state_mapping[i] for i in range(self.n_states)]
        return probs_df
    
    def generate_signals(
        self,
        regime_labels: np.ndarray,
        regime_probs: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate trading signals based on regime detection.
        
        Signal logic:
        - crowd_long regime with high confidence: consider short (fade the crowd)
        - crowd_short regime with high confidence: consider long (fade the crowd)
        - neutral: no signal
        
        Parameters
        ----------
        regime_labels : np.ndarray
            Regime labels for each observation
        regime_probs : pd.DataFrame
            Regime probabilities
        confidence_threshold : float
            Minimum probability to generate a signal
        
        Returns
        -------
        signals_df : pd.DataFrame
            DataFrame with signal, confidence, and regime info
        """
        n = len(regime_labels)
        signals = np.zeros(n, dtype=int)
        confidence = np.zeros(n)
        
        for i in range(n):
            regime = regime_labels[i]
            prob = regime_probs.iloc[i][regime]
            confidence[i] = prob
            
            if prob >= confidence_threshold:
                if regime == "crowd_long":
                    signals[i] = -1
                elif regime == "crowd_short":
                    signals[i] = 1
                else:
                    signals[i] = 0
        
        signals_df = pd.DataFrame({
            "regime": regime_labels,
            "confidence": confidence,
            "signal": signals,
            "crowd_long_prob": regime_probs["crowd_long"].values,
            "crowd_short_prob": regime_probs["crowd_short"].values
        })
        
        neutral_cols = [c for c in regime_probs.columns if "neutral" in c]
        if neutral_cols:
            signals_df["neutral_prob"] = regime_probs[neutral_cols].sum(axis=1).values
        else:
            signals_df["neutral_prob"] = 0.0
        
        return signals_df
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data under the model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")
        return self.model.score(X)
    
    def bic(self, X: np.ndarray) -> float:
        """
        Compute Bayesian Information Criterion.
        Lower is better.
        """
        n_samples, n_features = X.shape
        log_likelihood = self.score(X)
        n_params = self._count_params(n_features)
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X: np.ndarray) -> float:
        """
        Compute Akaike Information Criterion.
        Lower is better.
        """
        n_samples, n_features = X.shape
        log_likelihood = self.score(X)
        n_params = self._count_params(n_features)
        return -2 * log_likelihood + 2 * n_params
    
    def _count_params(self, n_features: int) -> int:
        """Count the number of free parameters in the model."""
        n = self.n_states
        
        n_start = n - 1
        n_trans = n * (n - 1)
        n_means = n * n_features
        
        if self.covariance_type == "full":
            n_covs = n * n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            n_covs = n * n_features
        elif self.covariance_type == "spherical":
            n_covs = n
        else:  # tied
            n_covs = n_features * (n_features + 1) // 2
        
        return n_start + n_trans + n_means + n_covs
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition probability matrix with regime labels.
        
        Returns
        -------
        trans_df : pd.DataFrame
            Labeled transition matrix
        """
        trans = self.model.transmat_
        labels = [self.state_mapping[i] for i in range(self.n_states)]
        return pd.DataFrame(trans, index=labels, columns=labels)
    
    def get_stationary_distribution(self) -> pd.Series:
        """
        Compute the stationary distribution of the Markov chain.
        
        Returns
        -------
        stationary : pd.Series
            Stationary probabilities for each regime
        """
        trans = self.model.transmat_
        eigenvalues, eigenvectors = np.linalg.eig(trans.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, stationary_idx])
        stationary = stationary / stationary.sum()
        
        labels = [self.state_mapping[i] for i in range(self.n_states)]
        return pd.Series(stationary, index=labels)
    
    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "FundingRegimeHMM":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def summary(self) -> None:
        """Print a summary of the fitted model."""
        if not self.fitted:
            print("Model not yet fitted")
            return
        
        print("=" * 60)
        print("FUNDING REGIME HMM SUMMARY")
        print("=" * 60)
        
        print(f"\nNumber of states: {self.n_states}")
        print(f"Covariance type: {self.model.covariance_type}")
        
        print("\n--- State Mapping ---")
        for state, label in self.state_mapping.items():
            print(f"  State {state}: {label}")
        
        print("\n--- Emission Means ---")
        means_df = pd.DataFrame(
            self.model.means_,
            index=[self.state_mapping[i] for i in range(self.n_states)],
            columns=self.feature_cols
        )
        print(means_df.round(3))
        
        print("\n--- Transition Matrix ---")
        print(self.get_transition_matrix().round(3))
        
        print("\n--- Stationary Distribution ---")
        print(self.get_stationary_distribution().round(3))
        
        print("\n--- Initial State Probabilities ---")
        start_probs = pd.Series(
            self.model.startprob_,
            index=[self.state_mapping[i] for i in range(self.n_states)]
        )
        print(start_probs.round(3))


def select_n_states(
    X: np.ndarray,
    n_states_range: List[int] = [2, 3, 4, 5],
    criterion: str = "bic",
    **kwargs
) -> Tuple[int, pd.DataFrame]:
    """
    Select optimal number of states using BIC or AIC.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    n_states_range : list
        List of n_states values to try
    criterion : str
        'bic' or 'aic'
    **kwargs
        Additional arguments passed to FundingRegimeHMM
    
    Returns
    -------
    best_n : int
        Optimal number of states
    results : pd.DataFrame
        DataFrame with scores for each n_states
    """
    results = []
    
    for n in n_states_range:
        try:
            model = FundingRegimeHMM(n_states=n, **kwargs)
            model.fit(X)
            
            ll = model.score(X)
            bic = model.bic(X)
            aic = model.aic(X)
            
            results.append({
                "n_states": n,
                "log_likelihood": ll,
                "bic": bic,
                "aic": aic
            })
        except Exception as e:
            print(f"Failed for n_states={n}: {e}")
    
    results_df = pd.DataFrame(results)
    
    if criterion == "bic":
        best_n = results_df.loc[results_df["bic"].idxmin(), "n_states"]
    else:
        best_n = results_df.loc[results_df["aic"].idxmin(), "n_states"]
    
    return int(best_n), results_df


def run_hmm_analysis(
    spot_path: str = "data/Price/btc-usdt-binance.csv",
    perp_path: str = "data/Price/btc-usdt-perp-binance.csv",
    funding_path: str = "data/Funding Rate/btc-usd-perp_binance.csv",
    n_states: int = 3,
    confidence_threshold: float = 0.6,
    standardize: bool = True,
    train_end_date: Optional[str] = None
) -> Tuple[FundingRegimeHMM, pd.DataFrame]:
    """
    Run complete HMM analysis pipeline.
    
    Parameters
    ----------
    spot_path : str
        Path to spot price data
    perp_path : str
        Path to perp price data
    funding_path : str
        Path to funding rate data
    n_states : int
        Number of hidden states
    confidence_threshold : float
        Minimum probability to generate a trading signal
    standardize : bool
        Whether to standardize features
    train_end_date : str, optional
        If provided, train only on data before this date (YYYY-MM-DD).
        Predictions are made on all data using the trained model.
    
    Returns
    -------
    model : FundingRegimeHMM
        Fitted HMM model
    results : pd.DataFrame
        DataFrame with timestamps, features, regimes, probabilities, and signals
    """
    X, features_df, scaling_params = prepare_hmm_input(
        spot_path, perp_path, funding_path, standardize
    )
    
    if train_end_date is not None:
        train_end = pd.to_datetime(train_end_date)
        train_mask = features_df["TIMESTAMP"] < train_end
        X_train = X[train_mask]
        print(f"Training on data before {train_end_date}: {train_mask.sum()} samples")
    else:
        X_train = X
    
    model = FundingRegimeHMM(n_states=n_states)
    model.fit(X_train)
    model.scaling_params = scaling_params
    
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    
    regime_labels = model.get_regime_labels(states)
    regime_probs = model.get_regime_probabilities(posteriors)
    
    signals = model.generate_signals(
        regime_labels, regime_probs, confidence_threshold
    )
    
    results = features_df.copy()
    results["state"] = states
    results["regime"] = regime_labels
    results["signal"] = signals["signal"].values
    results["confidence"] = signals["confidence"].values
    results["crowd_long_prob"] = signals["crowd_long_prob"].values
    results["crowd_short_prob"] = signals["crowd_short_prob"].values
    results["neutral_prob"] = signals["neutral_prob"].values
    
    if train_end_date is not None:
        results["is_train"] = results["TIMESTAMP"] < train_end
    
    return model, results


def run_walk_forward_hmm(
    spot_path: str = "data/Price/btc-usdt-binance.csv",
    perp_path: str = "data/Price/btc-usdt-perp-binance.csv",
    funding_path: str = "data/Funding Rate/btc-usd-perp_binance.csv",
    n_states: int = 3,
    train_months: int = 8,
    test_months: int = 1,
    expanding: bool = False,
    confidence_threshold: float = 0.6,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Run walk-forward HMM analysis with proper train/test splits.
    
    Parameters
    ----------
    train_months : int
        Training window in months (always 8 months rolling)
    test_months : int
        Test window (walk-forward step) in months
    expanding : bool
        If True, use expanding window. If False, use rolling window (default).
    
    Returns
    -------
    results : pd.DataFrame
        Combined results with walk-forward regime predictions
    """
    X_full, features_df, _ = prepare_hmm_input(
        spot_path, perp_path, funding_path, standardize=False
    )
    
    timestamps = features_df["TIMESTAMP"]
    start_date = timestamps.min()
    end_date = timestamps.max()
    
    train_start = start_date
    train_end = start_date + pd.DateOffset(months=train_months)
    
    all_results = []
    fold = 0
    
    while train_end < end_date:
        test_end = min(train_end + pd.DateOffset(months=test_months), end_date)
        
        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        test_mask = (timestamps >= train_end) & (timestamps < test_end)
        
        if test_mask.sum() == 0:
            break
        
        X_train = X_full[train_mask]
        X_test = X_full[test_mask]
        
        if standardize:
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            X_train_scaled = (X_train - mean) / std
            X_test_scaled = (X_test - mean) / std
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        model = FundingRegimeHMM(n_states=n_states)
        model.fit(X_train_scaled)
        
        states = model.predict(X_test_scaled)
        posteriors = model.predict_proba(X_test_scaled)
        regime_labels = model.get_regime_labels(states)
        regime_probs = model.get_regime_probabilities(posteriors)
        
        fold_results = features_df[test_mask].copy()
        fold_results["fold"] = fold
        fold_results["train_start"] = train_start
        fold_results["train_end"] = train_end
        fold_results["regime"] = regime_labels
        fold_results["crowd_long_prob"] = regime_probs["crowd_long"].values
        fold_results["crowd_short_prob"] = regime_probs["crowd_short"].values
        fold_results["neutral_prob"] = regime_probs["neutral"].values if "neutral" in regime_probs.columns else 0
        
        all_results.append(fold_results)
        
        print(f"Fold {fold}: Train {train_start.date()} to {train_end.date()}, "
              f"Test {train_end.date()} to {test_end.date()} ({test_mask.sum()} samples)")
        
        if expanding:
            pass
        else:
            train_start = train_start + pd.DateOffset(months=test_months)
        
        train_end = test_end
        fold += 1
    
    return pd.concat(all_results, ignore_index=True)


def fit_hmm_single_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_states: int = 3,
    confidence_threshold: float = 0.6,
    point_in_time: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit HMM on training data and predict on test data.
    For use within notebook walk-forward loop.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (already standardized)
    X_test : np.ndarray
        Test feature matrix (standardized with train params)
    n_states : int
        Number of HMM states
    confidence_threshold : float
        Minimum probability for regime signal
    point_in_time : bool
        If True, predict each timestamp using only data up to that point (no look-ahead).
        If False, use batch Viterbi on entire test set (uses future info within window).
    
    Returns
    -------
    regime_labels : np.ndarray
        Regime labels for test data
    crowd_long_prob : np.ndarray
        Crowd long probabilities
    crowd_short_prob : np.ndarray
        Crowd short probabilities
    neutral_prob : np.ndarray
        Neutral probabilities
    """
    model = FundingRegimeHMM(n_states=n_states)
    model.fit(X_train)
    
    if point_in_time:
        # Point-in-time prediction: predict each observation individually
        # using only data up to and including that timestamp
        n_test = len(X_test)
        states = np.zeros(n_test, dtype=int)
        posteriors = np.zeros((n_test, n_states))
        
        for t in range(n_test):
            # Use observations from start up to current timestamp
            X_up_to_t = X_test[:t+1]
            
            # Get posterior probabilities for current observation
            # predict_proba returns posteriors for all observations in sequence
            post_t = model.predict_proba(X_up_to_t)
            posteriors[t] = post_t[-1]  # Take last (current) timestamp
            states[t] = np.argmax(posteriors[t])
    else:
        # Batch prediction (original behavior - uses future info within test window)
        states = model.predict(X_test)
        posteriors = model.predict_proba(X_test)
    
    regime_labels = model.get_regime_labels(states)
    regime_probs = model.get_regime_probabilities(posteriors)
    
    crowd_long_prob = regime_probs["crowd_long"].values
    crowd_short_prob = regime_probs["crowd_short"].values
    neutral_prob = regime_probs["neutral"].values if "neutral" in regime_probs.columns else np.zeros(len(states))
    
    return regime_labels, crowd_long_prob, crowd_short_prob, neutral_prob


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HMM Regime Analysis")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Train only on data before this date (YYYY-MM-DD)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward analysis")
    parser.add_argument("--train-months", type=int, default=8,
                        help="Training window in months (for walk-forward)")
    parser.add_argument("--test-months", type=int, default=1,
                        help="Test window in months (for walk-forward)")
    args = parser.parse_args()
    
    print("Running HMM Regime Analysis...")
    print("-" * 60)
    
    if args.walk_forward:
        print(f"\nWalk-Forward Mode: {args.train_months}mo train, {args.test_months}mo test")
        print("=" * 60)
        
        results = run_walk_forward_hmm(
            train_months=args.train_months,
            test_months=args.test_months,
            expanding=True
        )
        
        print("\n--- Regime Distribution (Out-of-Sample) ---")
        print(results["regime"].value_counts(normalize=True).round(3))
        
        results.to_csv("hmm_results_wf.csv", index=False)
        print(f"\nWalk-forward results saved to hmm_results_wf.csv")
        
    else:
        n_states_to_use = 3
        
        if args.train_end:
            print(f"\nTraining on data before: {args.train_end}")
        else:
            print("\nWARNING: Training on ALL data (no train/test split)")
        
        print(f"Using n_states={n_states_to_use}")
        print("=" * 60)
        
        model, results = run_hmm_analysis(
            n_states=n_states_to_use,
            train_end_date=args.train_end
        )
        
        model.summary()
        
        print("\n--- Regime Distribution ---")
        print(results["regime"].value_counts(normalize=True).round(3))
        
        print("\n--- Signal Distribution ---")
        signal_map = {-1: "short", 0: "neutral", 1: "long"}
        print(results["signal"].map(signal_map).value_counts(normalize=True).round(3))
        
        print("\n--- Average Duration in Each Regime (hours) ---")
        trans = model.get_transition_matrix()
        for regime in trans.index:
            self_trans = trans.loc[regime, regime]
            avg_duration = 1 / (1 - self_trans) if self_trans < 1 else float('inf')
            print(f"  {regime}: {avg_duration:.1f}")
        
        results.to_csv("hmm_results.csv", index=False)
        print("\nResults saved to hmm_results.csv")
        
        model.save("hmm_model.pkl")
        print("Model saved to hmm_model.pkl")
