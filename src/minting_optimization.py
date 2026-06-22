"""
Minting optimization for bimetallic systems.
"""

import numpy as np
from scipy.optimize import minimize

class MintingOptimizer:
    """
    Optimal minting policy for gold and silver.
    """
    
    def __init__(self, model):
        self.model = model
    
    def objective(self, params):
        """
        Objective function: minimize minting costs.
        params = [alpha, delta] (gold fractions)
        """
        alpha, delta = params
        
        # Ensure constraints
        if alpha < 0 or alpha > 1 or delta < 0 or delta > 1:
            return 1e10
        
        # Run simulation with these parameters
        model_copy = self.model
        model_copy.alpha = alpha
        model_copy.delta = delta
        
        results = model_copy.simulate(T=50.0, dt=0.01)
        
        # Compute total minting costs
        total_mint = 0
        G = results['G']
        A = results['A']
        
        # Approximate minting from stock changes
        for i in range(1, len(G)):
            dG = G[i] - G[i-1]
            dA = A[i] - A[i-1]
            if dG < 0:
                total_mint += abs(dG) * model_copy.kappa_G
            if dA < 0:
                total_mint += abs(dA) * model_copy.kappa_A
        
        # Penalty for stock depletion
        if G[-1] < 50 or A[-1] < 500:
            total_mint += 1e6
        
        return total_mint
    
    def optimize(self):
        """
        Find optimal alpha and delta.
        
        Returns:
            Optimal parameters and optimization result
        """
        # Initial guess
        x0 = [0.5, 0.5]
        
        # Bounds
        bounds = [(0.1, 0.9), (0.1, 0.9)]
        
        # Optimization
        result = minimize(
            self.objective,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 100, 'xatol': 0.01}
        )
        
        return result.x, result
    
    def evaluate_alpha_delta_space(self, n_points=20):
        """
        Evaluate objective over alpha-delta space for visualization.
        
        Returns:
            Grid of alpha, delta, and objective values
        """
        alpha_vals = np.linspace(0.1, 0.9, n_points)
        delta_vals = np.linspace(0.1, 0.9, n_points)
        
        Z = np.zeros((n_points, n_points))
        
        for i, alpha in enumerate(alpha_vals):
            for j, delta in enumerate(delta_vals):
                self.model.alpha = alpha
                self.model.delta = delta
                Z[i, j] = self.objective([alpha, delta])
        
        return alpha_vals, delta_vals, Z
