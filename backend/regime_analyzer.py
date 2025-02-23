import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from geometric_regime_analyzer import GeometricRegimePredictor
import yfinance as yf
from datetime import datetime, timedelta
from copy import deepcopy

logger = logging.getLogger(__name__)

class RegimeAnalyzer:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.vol_threshold = 0.015
        self.geometric_predictor = GeometricRegimePredictor()
        
        # Initialize HMM with better parameters
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,  # Increased from 100
            tol=1e-5,    # Added tolerance parameter
            random_state=42,
            init_params="kmeans"  # Better initialization - kmeans clustering mogs
        )
        
        # Initialize with more stable starting values
        self.hmm_model.startprob_ = np.array([0.6, 0.3, 0.1])  # Bull, Bear, Transition
        self.hmm_model.transmat_ = np.array([
            [0.98, 0.01, 0.01],  # Bull state transitions
            [0.01, 0.98, 0.01],  # Bear state transitions
            [0.45, 0.45, 0.10]   # Transition state transitions
        ])
        
        # Set initial means and covariances with more realistic values
        self.hmm_model.means_ = np.array([[0.001], [-0.001], [0]])
        self.hmm_model.covars_ = np.array([
            [[0.0002]],  # Bull state variance
            [[0.0004]],  # Bear state variance
            [[0.0003]]   # Transition state variance
        ])
        
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def _train_hmm(self):
        try:
            # Get training data
            training_data = yf.download(
                'SPY',
                start=(datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d'),
                progress=False
            )
            
            # Calculate returns and prepare for HMM
            returns = training_data['Close'].pct_change().dropna().values.reshape(-1, 1)
            
            # Scale returns to help with convergence
            scaled_returns = returns * 100  # Convert to percentage
            
            # Try multiple initializations
            best_score = float('-inf')
            best_model = None
            
            for _ in range(3):  # Try 3 different initializations?
                try:
                    self.hmm_model.fit(scaled_returns)
                    score = self.hmm_model.score(scaled_returns)
                    
                    if score > best_score:
                        best_score = score
                        best_model = deepcopy(self.hmm_model)
                except Exception as e:
                    logger.warning(f"HMM fitting attempt failed: {str(e)}")
                    continue
            
            if best_model is not None:
                self.hmm_model = best_model
                logger.info("HMM model successfully trained")
            else:
                logger.warning("Using default HMM parameters as training failed")
            
        except Exception as e:
            logger.error(f"Error in HMM training: {str(e)}")
            logger.info("Using default HMM parameters")

    def _init_and_fit_hmm(self, features):
        try:
            self.hmm_model.fit(features)
            self.is_fitted = True
            return True
        except Exception as e:
            logger.error(f"Error fitting HMM: {str(e)}")
            return False

    def prepare_features(self, curvature_history: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        try:
            # First, ensure indexes match
            common_index = curvature_history.index.intersection(returns.index)
            curvature_history = curvature_history.loc[common_index]
            returns = returns.loc[common_index]

            # Create features DataFrame
            features = pd.concat([
                curvature_history.mean(axis=1),  # Average curvature
                curvature_history.std(axis=1),   # Curvature dispersion
                returns.mean(axis=1),            # Market return
                returns.std(axis=1),             # Market volatility
                curvature_history.quantile(0.1, axis=1),  # Lower tail behavior
                curvature_history.quantile(0.9, axis=1),  # Upper tail behavior
                curvature_history.rolling(5).mean().mean(axis=1),  # 5-day trend
            ], axis=1)

            # Handle NaN values
            features = features.fillna(method='ffill')  # Forward fill
            features = features.fillna(method='bfill')  # Backward fill for any remaining NaNs
            
            # Drop any remaining rows with NaN values
            features = features.dropna()

            if len(features) == 0:
                raise ValueError("No valid data points after cleaning")

            # Scale the features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features

        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            raise ValueError(f"Failed to prepare features: {str(e)}")

    def detect_regimes(self, curvature_history: pd.DataFrame, returns: pd.DataFrame):
        try:
            # Calculate time span
            time_span = (returns.index[-1] - returns.index[0]).days
            logger.info(f"Analyzing time span of {time_span} days")

            # For very short periods (less than 5 days)
            if len(returns) < 5:  # Check number of data points instead of days
                logger.info("Short timeframe detected, using simplified regime detection")
                
                # Calculate simple metrics
                daily_ret = returns.mean()
                daily_vol = returns.std()
                
                # Simple regime determination
                avg_ret = daily_ret.mean()
                avg_vol = daily_vol.mean()
                
                # Determine regime type based on return and volatility
                if avg_ret > 0:
                    regime_type = 'stable_bull' if avg_vol < 0.02 else 'volatile_bull'
                else:
                    regime_type = 'stable_bear' if avg_vol < 0.02 else 'volatile_bear'
                
                # Create regime stats
                regime_stats = {
                    0: {
                        'avg_return': float(avg_ret * 100),  # Convert to percentage
                        'volatility': float(avg_vol * 100),  # Convert to percentage
                        'duration': len(returns),  # Use number of data points
                        'stability': 1.0  # Maximum stability for short period
                    }
                }
                
                logger.info(f"Short-term regime detected: {regime_type}")
                
                return {
                    'regimes': [0] * len(returns),
                    'regime_stats': regime_stats,
                    'regime_types': {0: regime_type},
                    'transition_matrix': [[1.0]],  # No transitions for short period
                    'stability': [1.0],
                    'dates': returns.index.strftime('%Y-%m-%d').tolist()
                }
                
            # For longer periods, use the HMM approach
            else:
                # Original regime detection logic for longer periods
                # Ensure we have enough data
                if len(curvature_history) < 30 or len(returns) < 30:
                    logger.error("Not enough data for HMM regime detection")
                    raise ValueError("Insufficient data for regime detection (minimum 30 data points required)")

                # Remove any NaN values
                curvature_history = curvature_history.fillna(method='ffill').fillna(method='bfill')
                returns = returns.fillna(method='ffill').fillna(method='bfill')

                # Prepare features
                features = self.prepare_features(curvature_history, returns)
                
                if len(features) == 0:
                    raise ValueError("Failed to prepare features for regime detection")

                # Fit HMM and predict regimes
                try:
                    self.hmm_model.fit(features)
                except Exception as e:
                    logger.error(f"Error fitting HMM model: {str(e)}")
                    return None
                
                regimes = self.hmm_model.predict(features)
                
                # Calculate regime characteristics
                regime_stats = {}
                for i in range(self.n_regimes):
                    mask = (regimes == i)
                    if np.any(mask):  # Only calculate stats if we have data points for this regime
                        regime_stats[i] = {
                            'avg_curvature': float(np.mean(features[mask, 0])),
                            'avg_dispersion': float(np.mean(features[mask, 1])),
                            'avg_return': float(np.mean(features[mask, 2])),
                            'volatility': float(np.std(features[mask, 2])),
                            'duration': int(np.sum(mask)),
                            'transitions': int(np.sum(np.diff(regimes) == i))
                        }
                    else:
                        logger.warning(f"No data points found for regime {i}")
                        regime_stats[i] = {
                            'avg_curvature': 0.0,
                            'avg_dispersion': 0.0,
                            'avg_return': 0.0,
                            'volatility': 0.0,
                            'duration': 0,
                            'transitions': 0
                        }
                
                # Classify regimes
                regime_types = self._classify_regimes(regime_stats)
                
                # Get transition probabilities
                transitions = self.hmm_model.transmat_
                
                # Calculate regime stability
                stability = np.diag(transitions)
                
                # Convert dates to string format for JSON serialization
                dates = curvature_history.index.strftime('%Y-%m-%d').tolist()
                
                return {
                    'regimes': regimes.tolist(),
                    'regime_stats': regime_stats,
                    'regime_types': regime_types,
                    'transition_matrix': transitions.tolist(),
                    'stability': stability.tolist(),
                    'dates': dates
                }

        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            raise ValueError(f"Failed to detect market regimes: {str(e)}")

    def _classify_regimes(self, regime_stats: Dict) -> Dict[int, str]:
        classifications = {}
        
        for regime_id, stats in regime_stats.items():
            ret = stats['avg_return']  # Already in percentage
            vol = stats['volatility']  # Already in percentage
            
            # Classification based on daily metrics
            if ret > 0:
                if vol < 1.0:  # 1% daily volatility threshold
                    classifications[regime_id] = 'stable_bull'
                else:
                    classifications[regime_id] = 'volatile_bull'
            else:
                if vol < 1.0:
                    classifications[regime_id] = 'stable_bear'
                else:
                    classifications[regime_id] = 'volatile_bear'
        
        return classifications

    def get_current_regime(self, 
                          current_features: np.ndarray, 
                          regime_history: List[int]) -> Dict:
        try:
            # Get current regime
            current_regime = self.hmm_model.predict([current_features])[-1]
            
            # Get transition probabilities for current regime
            transition_probs = self.hmm_model.transmat_[current_regime]
            
            # Calculate regime duration
            current_duration = 1
            for i in reversed(regime_history[:-1]):
                if i == current_regime:
                    current_duration += 1
                else:
                    break
            
            return {
                'current_regime': int(current_regime),
                'duration': current_duration,
                'transition_probs': transition_probs.tolist(),
                'stability': float(transition_probs[current_regime])
            }
            
        except Exception as e:
            logger.error(f"Error in current regime analysis: {str(e)}")
            return None

    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        try:
            return self.hmm_model.predict_proba(features)
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {str(e)}")
            return None

    def analyze_regime(self, sector_indices: pd.DataFrame, returns: pd.DataFrame, curvature_history: pd.DataFrame, timeframe: int = 7) -> Optional[Dict]:
        try:
            logger.info(f"Starting regime analysis with timeframe: {timeframe} days")
            
            # Calculate market return for HMM
            market_return = returns.mean(axis=1).values.reshape(-1, 1)
            logger.info(f"Market return shape: {market_return.shape}")
            
            # Calculate regime probabilities using HMM
            regime_probs = self.hmm_model.predict_proba(market_return)[-1]
            logger.info(f"Raw regime probabilities: {regime_probs}")
            
            # Calculate market metrics
            breadth_signal = self._calculate_market_breadth(returns)
            recent_vol = returns.std().mean()
            avg_correlation = returns.corr().mean().mean()
            logger.info(f"Market metrics - Breadth: {breadth_signal:.3f}, Vol: {recent_vol:.3f}, Corr: {avg_correlation:.3f}")
            
            # Calculate confidence components with detailed logging
            prob_spread = max(regime_probs) - np.median(regime_probs)
            regime_certainty = abs(regime_probs[0] - regime_probs[1]) / (regime_probs[0] + regime_probs[1])
            logger.info(f"Probability spread: {prob_spread:.3f}, Regime certainty: {regime_certainty:.3f}")
            
            breadth_alignment = 1 if (breadth_signal > 0) == (regime_probs[0] > regime_probs[1]) else -1
            vol_regime_alignment = 1 if (recent_vol > self.vol_threshold) == (regime_probs[1] > regime_probs[0]) else -1
            corr_regime_alignment = 1 if (avg_correlation > 0.6) == (regime_probs[1] > regime_probs[0]) else -1
            logger.info(f"Alignments - Breadth: {breadth_alignment}, Vol: {vol_regime_alignment}, Corr: {corr_regime_alignment}")
            
            # Dynamic base confidence
            base_confidence = 0.15 + (0.15 * regime_certainty)
            logger.info(f"Base confidence: {base_confidence:.3f}")
            
            # Time decay factor
            time_decay = max(1.0 - (timeframe / 60), 0.5)
            logger.info(f"Time decay factor: {time_decay:.3f}")
            
            # Calculate raw confidence with absolute values and different weights
            raw_confidence = base_confidence + (
                0.3 * prob_spread +  # More weight on probability spread
                0.2 * abs(breadth_signal) +  # Use absolute value instead of alignment
                0.2 * (1 - recent_vol/self.vol_threshold) +  # Lower volatility = higher confidence
                0.3 * abs(avg_correlation - 0.5)  # Distance from neutral correlation
            )
            logger.info(f"Raw confidence before time decay: {raw_confidence:.3f}")
            
            # Apply time decay
            confidence = raw_confidence * time_decay
            logger.info(f"Confidence after time decay: {confidence:.3f}")
            
            # Scale confidence to [0.15,0.85]
            confidence = 0.15 + 0.7 * min(confidence, 1.0)  # Removed the max(0) since we're using absolutes
            logger.info(f"Final scaled confidence: {confidence:.3f}")
            
            # Adjust regime probabilities based on timeframe
            time_uncertainty = min(timeframe / 60, 0.5)  # Max 50% uncertainty from time
            adjusted_probs = regime_probs.copy()
            
            # Move probabilities closer to uncertainty as timeframe increases
            adjusted_probs[0] = regime_probs[0] * (1 - time_uncertainty) + 0.33 * time_uncertainty
            adjusted_probs[1] = regime_probs[1] * (1 - time_uncertainty) + 0.33 * time_uncertainty
            adjusted_probs[2] = regime_probs[2] * (1 - time_uncertainty) + 0.33 * time_uncertainty
            
            # Normalize probabilities
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            # Calculate transition probability with time consideration
            regime_uncertainty = 1 - abs(adjusted_probs[0] - adjusted_probs[1])
            base_transition = 0.3 + (0.3 * regime_uncertainty)  # Base between 30-60%
            
            time_factor = timeframe / 30  # Scales linearly with prediction timeframe
            vol_impact = min(recent_vol / self.vol_threshold, 1.5)
            
            transition_prob = min(
                base_transition + 
                (0.2 * time_factor) +  # Time effect increases with longer predictions
                (0.15 * vol_impact) +  # Volatility effect
                (0.1 * (1 - confidence)) +  # Lower confidence increases transition probability
                (0.1 * regime_uncertainty),  # More uncertainty means higher transition chance
                0.85  # Cap at 85%
            )
            
            # Get geometric prediction with sector movements
            geometric_prediction = self.geometric_predictor.predict_regime_transition(
                sector_indices, 
                "BEAR MARKET" if adjusted_probs[1] > adjusted_probs[0] else "BULL MARKET",
                timeframe
            )
            
            if geometric_prediction is None:
                raise ValueError("Failed to generate geometric prediction")
            
            # Create response with adjusted probabilities
            response_data = {
                'predicted_regime': "BEAR MARKET" if adjusted_probs[1] > adjusted_probs[0] else "BULL MARKET",
                'regime_probabilities': {
                    'Bull Market': float(adjusted_probs[0] * 100),
                    'Bear Market': float(adjusted_probs[1] * 100),
                    'Transition': float(adjusted_probs[2] * 100)
                },
                'confidence_metrics': {
                    'model_confidence': float(confidence * 100),
                    'transition_probability': float(transition_prob * 100),
                    'geometric_confidence': float(geometric_prediction['confidence'] * 100)
                },
                'sector_movements': geometric_prediction['sector_movements']
            }
            
            logger.info(f"Generated response with sector movements: {response_data['sector_movements']}")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def _combine_predictions(self,
                           hmm_probs: np.ndarray,
                           geometric_pred: Dict) -> Dict:
        # Weight factors
        w_hmm = 0.6
        w_geometric = 0.4
        
        # Combined probability
        combined_probs = {}
        for i, regime_type in enumerate(['stable_bull', 'stable_bear', 'volatile']):
            hmm_prob = hmm_probs[i]
            geo_prob = 1.0 if geometric_pred['predicted_regime'] == regime_type else 0.0
            combined_probs[regime_type] = w_hmm * hmm_prob + w_geometric * geo_prob
            
        # Get most likely regime
        predicted_regime = max(combined_probs.items(), key=lambda x: x[1])[0]
        
        return {
            'regime': predicted_regime,
            'probabilities': combined_probs,
            'confidence': geometric_pred['confidence']
        }
        
    def _compute_confidence_metrics(self,
                                  hmm_probs: np.ndarray,
                                  geometric_pred: Dict) -> Dict:
        return {
            'hmm_confidence': float(np.max(hmm_probs)),
            'geometric_confidence': geometric_pred['confidence'],
            'entropy': geometric_pred['entropy'],
            'transition_probability': geometric_pred['transition_probability']
        }

    def get_regime_type(self, features: np.ndarray) -> str:
        # Calculate basic statistics from features
        avg_return = np.mean(features[:, 0])  # First column is returns mean
        volatility = np.mean(features[:, 1])  # Second column is returns std
        
        # Basic regime classification based on return and volatility
        if volatility > 0.02:  # High volatility threshold
            return 'volatile'
        elif avg_return > 0.001:  # Positive trend threshold
            return 'trending'
        else:
            return 'normal'

    def _calculate_market_breadth(self, returns: pd.DataFrame) -> float:
        try:
            # Calculate the percentage of sectors with positive returns
            recent_returns = returns.iloc[-5:]  # Last 5 days
            advancing_sectors = (recent_returns > 0).mean()
            
            # Calculate breadth as 2*(percentage advancing) - 1
            # This maps [0,1] to [-1,1]
            breadth = 2 * advancing_sectors.mean() - 1
            
            # Add momentum component
            momentum = returns.mean().mean()  # Average return across all sectors
            
            # Combine breadth and momentum
            composite_breadth = 0.7 * breadth + 0.3 * np.sign(momentum)
            
            logger.info(f"Calculated market breadth: {composite_breadth}")
            return float(composite_breadth)
            
        except Exception as e:
            logger.error(f"Error calculating market breadth: {str(e)}")
            return 0.0  # Default to neutral if calculation fails

    def _calculate_transition_probability(self, returns: pd.DataFrame, regime_probs: np.ndarray, volatility: float, correlation: float) -> float:
        try:
            # Get current most likely regime
            current_regime = np.argmax(regime_probs)
            
            # Base transition probability from HMM transition matrix
            base_prob = 1 - self.hmm_model.transmat_[current_regime][current_regime]
            
            # Volatility factor: higher volatility increases transition probability
            vol_factor = min(volatility / self.vol_threshold, 2.0)
            
            # Correlation factor: extreme correlations (high or low) increase transition probability
            corr_factor = 2 * abs(correlation - 0.5)  # Maps [0,1] to [0,1] with peak at extremes
            
            # Regime probability factor: less confident regime predictions increase transition probability
            regime_confidence = max(regime_probs) - np.median(regime_probs)
            prob_factor = 1 - regime_confidence
            
            # Combine factors
            transition_prob = base_prob * (
                0.4 * vol_factor +
                0.3 * corr_factor +
                0.3 * prob_factor
            )
            
            # Scale to [0,1]
            transition_prob = max(min(transition_prob, 1.0), 0.0)
            
            logger.info(f"Calculated transition probability: {transition_prob}")
            return float(transition_prob)
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {str(e)}")
            return 0.5  # Default to moderate probability if calculation fails 

class GeometricRegimePredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict_regime_transition(self, sector_indices: pd.DataFrame, current_regime: str, timeframe: int) -> Dict:
        try:
            # Calculate recent sector performance
            returns = sector_indices.pct_change()
            recent_performance = returns.tail(20).mean()
            volatility = returns.tail(20).std()
            
            # Calculate momentum with more weight on recent data
            momentum = returns.tail(5).mean() * 0.7 + returns.tail(20).mean() * 0.3
            relative_strength = recent_performance / volatility
            
            # Scale factor for timeframe (more aggressive)
            time_scale = (timeframe / 7) ** 0.8  # Non-linear scaling
            
            sector_movements = {}
            
            # Dynamic max move based on timeframe and regime
            base_max = 2.0 if timeframe <= 7 else (3.5 if timeframe <= 14 else 5.0)
            max_move = base_max * (1.2 if current_regime == "BEAR MARKET" else 1.0)
            
            for sector in sector_indices.columns:
                # Sector-specific characteristics
                is_tech = sector in ['Technology', 'Communication']
                is_financial = sector in ['Financial', 'RealEstate']
                is_defensive = sector in ['ConsumerDefensive', 'Healthcare', 'Utilities']
                
                sector_vol = volatility[sector]
                sector_momentum = momentum[sector]
                sector_strength = relative_strength[sector]
                
                if current_regime == "BEAR MARKET":
                    # Stronger sector-specific impacts
                    tech_impact = 0.04 if is_tech else 0  # Doubled
                    fin_impact = 0.03 if is_financial else 0  # Doubled
                    def_bonus = 0.02 if is_defensive else 0
                    
                    base_move = -1.5  # Base negative move in bear market
                    expected_move = (
                        base_move +
                        -0.5 * sector_momentum +
                        -0.5 * sector_vol +
                        -tech_impact +
                        -fin_impact +
                        def_bonus
                    ) * time_scale
                else:
                    # Bull market characteristics
                    tech_bonus = 0.03 if is_tech else 0
                    fin_bonus = 0.02 if is_financial else 0
                    def_impact = -0.01 if is_defensive else 0
                    
                    base_move = 1.0  # Base positive move in bull market
                    expected_move = (
                        base_move +
                        0.5 * sector_momentum +
                        0.3 * sector_strength +
                        tech_bonus + fin_bonus + def_impact
                    ) * time_scale
                
                # Scale movements based on timeframe
                scaled_move = np.clip(expected_move, -max_move, max_move)
                
                # Add sector-specific variation
                variation = 0.2 * sector_vol * time_scale
                final_move = scaled_move * (1 + variation)
                
                sector_movements[sector] = {
                    'direction': 'up' if final_move > 0 else 'down',
                    'magnitude': abs(float(final_move)),
                    'confidence': float(min(
                        0.3 + abs(sector_strength) * 0.4,
                        0.7
                    ))
                }
            
            return {
                'sector_movements': sector_movements,
                'confidence': min(0.7, 0.3 + abs(momentum.mean()) * 0.4)
            }
            
        except Exception as e:
            logger.error(f"Error in geometric prediction: {str(e)}")
            return None 