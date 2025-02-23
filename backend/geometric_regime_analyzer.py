import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GeometricRegimePredictor:
    def __init__(self, n_sectors: int = 11):
        self.n_sectors = n_sectors
        self.manifold_memory = {}
        self.sector_centroids = {}
        self.memory_length = 60  # Days to keep in memory
        self.transition_threshold = 0.7
        self.entropy_threshold = 0.5
        
    def compute_christoffel_symbols(self, metric: np.ndarray) -> np.ndarray:

        n = metric.shape[0]
        # Initialize Christoffel symbols array
        christoffel = np.zeros((n, n, n))
        
        try:
            # Compute inverse metric (with safety checks)
            metric_inv = np.linalg.pinv(metric)  # Using pseudoinverse for stability
            
            # Compute derivatives (simplified version)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        # Simplified derivative calculation
                        christoffel[i,j,k] = 0.5 * metric_inv[i,k] * (
                            metric[j,k] + metric[k,j] - metric[i,j]
                        )
            
            return christoffel
            
        except Exception as e:
            logger.error(f"Error in Christoffel computation: {e}")
            return np.zeros((n, n, n))

    def compute_sectoral_ricci(self, sector_data: pd.DataFrame, window: int = 30) -> np.ndarray:
        try:
            # Get dimensions
            n_sectors = len(sector_data.columns)
            
            # Initialize Ricci tensor
            ricci = np.zeros((n_sectors, n_sectors))
            
            # Compute returns
            returns = sector_data.pct_change().dropna()
            if len(returns) == 0:
                logger.error("No valid returns for Ricci computation")
                return ricci
            
            # Compute correlation matrix
            corr_matrix = returns.corr().fillna(0).values
            
            # Convert correlations to distances (with safety checks)
            distances = np.clip(1 - corr_matrix, 0, 2)  # Ensure values are in [0,2]
            metric = np.sqrt(distances + 1e-10)  # Add small constant to avoid sqrt(0)
            
            # Compute Christoffel symbols
            christoffel = self.compute_christoffel_symbols(metric)
            
            # Compute Ricci tensor components
            for i in range(n_sectors):
                for j in range(n_sectors):
                    ricci[i,j] = np.sum([
                        christoffel[k,i,l] * christoffel[l,j,k] -
                        christoffel[k,i,j] * christoffel[l,k,l]
                        for k in range(n_sectors)
                        for l in range(n_sectors)
                    ])
            
            return ricci
            
        except Exception as e:
            logger.error(f"Error in Ricci computation: {e}")
            return np.zeros((n_sectors, n_sectors))

    def integrate_geodesic_flow(self,
                              initial_positions: Dict[str, np.ndarray],
                              velocity_field: np.ndarray,
                              timesteps: int = 30) -> Dict[str, np.ndarray]:
        def geodesic_equation(t, state, Γ, velocity):
            # Split state into position and velocity components
            x, v = state[:self.n_sectors], state[self.n_sectors:]
            
            # Geodesic equation
            dxdt = v
            dvdt = -np.sum(Γ[..., np.newaxis] * v[np.newaxis, :] * v[:, np.newaxis], 
                          axis=(0, 1)) + velocity
            
            return np.concatenate([dxdt, dvdt])
        
        # Compute Christoffel symbols
        metric = self.compute_metric_from_velocity(velocity_field)
        Γ = self.compute_christoffel_symbols(metric)
        
        future_positions = {}
        for sector, pos in initial_positions.items():
            # Initial state: position and velocity
            initial_state = np.concatenate([pos, velocity_field[0]])
            
            # Solve geodesic equation
            solution = solve_ivp(
                fun=lambda t, y: geodesic_equation(t, y, Γ, velocity_field),
                t_span=(0, timesteps),
                y0=initial_state,
                method='RK45',
                t_eval=np.arange(timesteps)
            )
            
            future_positions[sector] = solution.y[:self.n_sectors, :]
            
        return future_positions

    def compute_manifold_entropy(self, state: Dict) -> float:
        if 'ricci_tensor' not in state:
            return 0.0
            
        # Eigenvalues of Ricci tensor
        eigenvals = np.linalg.eigvals(state['ricci_tensor'])
        
        # Normalize eigenvalues
        normalized_eigenvals = eigenvals / np.abs(eigenvals).max()
        
        # Compute entropy
        entropy = -np.sum(np.abs(normalized_eigenvals) * 
                         np.log(np.abs(normalized_eigenvals) + 1e-10))
        
        return entropy

    def find_flow_critical_points(self, state: Dict) -> List[Tuple[float, float]]:
        if 'ricci_tensor' not in state:
            return []
            
        # Eigendecomposition of Ricci tensor
        eigenvals, eigenvecs = np.linalg.eigh(state['ricci_tensor'])
        
        # Find points where eigenvalues change sign
        critical_points = []
        for i, (eval, evec) in enumerate(zip(eigenvals, eigenvecs.T)):
            if np.abs(eval) < 1e-6:  # Near zero
                critical_points.append((float(eval), evec))
                
        return critical_points

    def predict_regime_transition(self, sector_indices: pd.DataFrame, current_regime: str, timeframe: int = 7) -> Optional[Dict]:
        try:
            # Calculate inter-sector correlations
            returns = sector_indices.pct_change().fillna(0)
            correlations = returns.corr()
            
            # Calculate sector-specific metrics
            sector_metrics = {}
            for sector in returns.columns:
                # Calculate momentum
                short_momentum = returns[sector].tail(timeframe).mean()
                long_momentum = returns[sector].tail(timeframe * 2).mean()
                
                # Calculate relative strength
                sector_vol = returns[sector].std()
                relative_strength = short_momentum / (sector_vol + 1e-6)
                
                # Calculate correlation with other sectors
                avg_correlation = correlations[sector].mean()
                
                sector_metrics[sector] = {
                    'momentum': short_momentum,
                    'relative_strength': relative_strength,
                    'correlation': avg_correlation
                }
            
            # Calculate sector movements
            sector_movements = {}
            for sector in returns.columns:
                metrics = sector_metrics[sector]
                
                # Base movement calculation
                base_movement = (
                    metrics['momentum'] * 0.4 +
                    metrics['relative_strength'] * 0.4 +
                    (1 - metrics['correlation']) * 0.2
                )
                
                # Adjust for timeframe
                time_factor = np.sqrt(timeframe / 7)
                movement = base_movement * time_factor
                
                # Sector-specific adjustments
                if current_regime == "bull_market":
                    if sector in ['Technology', 'Financial', 'Consumer Cyclical']:
                        movement *= 1.2
                    elif sector in ['Utilities', 'Consumer Defensive']:
                        movement *= 0.8
                elif current_regime == "bear_market":
                    if sector in ['Utilities', 'Consumer Defensive']:
                        movement *= 1.2
                    elif sector in ['Technology', 'Financial', 'Consumer Cyclical']:
                        movement *= 0.8
                
                sector_movements[sector] = float(movement)
            
            # Calculate confidence based on prediction consistency
            movement_consistency = 1 - np.std(list(sector_movements.values()))
            correlation_penalty = np.mean([m['correlation'] for m in sector_metrics.values()])
            base_confidence = 0.7 * movement_consistency + 0.3 * (1 - correlation_penalty)
            
            # Adjust confidence for timeframe
            time_discount = np.sqrt(7 / timeframe)
            confidence = max(0.35, min(0.85, base_confidence * time_discount))
            
            return {
                "predicted_regime": current_regime,
                "confidence": float(confidence),
                "sector_movements": sector_movements,
                "sector_metrics": sector_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in geometric prediction: {str(e)}")
            return None

    def _analyze_market_state(self, returns: pd.DataFrame, timeframe: int) -> Dict:
        try:
            # Calculate rolling metrics
            window = min(timeframe * 5, 30)  # Use adaptive window size
            
            # Convert pandas Series to float using mean()
            rolling_vol = returns.rolling(window=window).std().mean().mean()  # Added .mean()
            rolling_mean = returns.rolling(window=window).mean().mean().mean()  # Added .mean()
            
            # Calculate cross-sectional volatility
            cross_vol = returns.std(axis=1).mean()  # This is already a float
            
            return {
                'volatility': float(cross_vol),
                'trend': float(rolling_mean),
                'rolling_volatility': float(rolling_vol)
            }
        except Exception as e:
            logger.error(f"Error in market state analysis: {str(e)}")
            # Return default values if calculation fails
            return {
                'volatility': 0.02,
                'trend': 0.0,
                'rolling_volatility': 0.02
            }

    def _predict_next_regime(self,
                            sector_movements: Dict,
                            current_regime: str,
                            market_volatility: float,
                            timeframe: int) -> Tuple[str, float]:
        movements = np.array(list(sector_movements.values()))
        avg_movement = np.mean(movements)
        dispersion = np.std(movements)
        
        # Adjust thresholds based on timeframe
        threshold_scale = np.sqrt(timeframe / 7)
        base_threshold = 0.5 / threshold_scale  # Threshold decreases with longer timeframes
        
        # Scale confidence based on timeframe
        time_scale = 1 / np.sqrt(timeframe / 7)  # Confidence decreases with time
        vol_scale = 1 / (1 + market_volatility)
        base_confidence = 0.56
        
        # Adjust confidence more aggressively for longer timeframes
        confidence = max(0.3, min(0.9, base_confidence * time_scale * vol_scale))
        
        # More sensitive regime determination for longer timeframes
        if avg_movement > dispersion * base_threshold:
            next_regime = "bull_market"
        elif avg_movement < -dispersion * base_threshold:
            next_regime = "bear_market"
        else:
            # For longer timeframes, favor transition states
            if timeframe > 14:
                next_regime = "transition"
            else:
                next_regime = "stable_" + current_regime.split("_")[0]
        
        # Reduce confidence more for longer timeframes
        if timeframe > 14:
            confidence *= 0.8
        if timeframe > 21:
            confidence *= 0.9
        
        return next_regime, confidence

    def _compute_transition_probability(self,
                                     entropy: float,
                                     critical_points: List,
                                     current_state: Dict) -> float:
        # Base probability from entropy
        base_prob = min(entropy / self.entropy_threshold, 1.0)
        
        # Adjust for critical points
        critical_point_factor = len(critical_points) * 0.1
        
        # Adjust for Ricci scalar
        R = np.trace(current_state['ricci_tensor'])
        curvature_factor = np.abs(R) / (1 + np.abs(R))  # Bounded in [0,1]
        
        # Combine factors
        transition_prob = (0.4 * base_prob + 
                         0.3 * critical_point_factor +
                         0.3 * curvature_factor)
        
        return min(transition_prob, 1.0)

    def analyze_sector_flow(self, sector_data: pd.DataFrame) -> Dict:
        # Compute sectoral Ricci tensor
        ricci = self.compute_sectoral_ricci(sector_data)
        
        # Calculate sectoral velocity field (from Ricci flow)
        velocity = -2 * ricci
        
        # Update sector centroids (using returns as proxy for position)
        returns = sector_data.pct_change().dropna()
        for sector in sector_data.columns:
            self.sector_centroids[sector] = returns[sector].iloc[-30:].values
            
        # Predict future positions
        future_positions = self.integrate_geodesic_flow(
            self.sector_centroids,
            velocity,
            timesteps=30
        )
        
        # Compute flow strength and stress
        flow_strength = np.linalg.norm(velocity, axis=(0,1))
        sectoral_stress = np.diagonal(ricci)  # Use diagonal elements as stress indicators
        
        return {
            'predicted_movements': future_positions,
            'flow_strength': float(flow_strength),
            'sectoral_stress': sectoral_stress.tolist()
        }

    def compute_metric_from_velocity(self, velocity_field: np.ndarray) -> np.ndarray:
        n = velocity_field.shape[0]
        metric = np.zeros((n, n))
        
        # Simple metric based on velocity correlations
        for i in range(n):
            for j in range(n):
                metric[i,j] = np.dot(velocity_field[i], velocity_field[j])
                
        # Ensure metric is positive definite
        metric = metric + np.eye(n) * 1e-6
        
        return metric 