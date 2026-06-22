"""
Bimetallic Monetary System (Nuqud) Model.
Implements gold (dinar) and silver (dirham) dynamics with the Yusufian cycle.
"""

import numpy as np
from scipy.integrate import odeint

class NuqudModel:
    """
    Bimetallic monetary system with gold and silver.
    
    State variables:
        S(t): Stock of essential goods
        G(t): Gold stock (dinar, ounces)
        A(t): Silver stock (dirham, ounces)
    
    Dynamics:
        dS/dt = P(t) - C(t)
        dG/dt = alpha*(P-C)_+ - delta*C - kappa_G*Mint_G
        dA/dt = (1-alpha)*(P-C)_+ - (1-delta)*C - kappa_A*Mint_A
    """
    
    def __init__(self, eta=0.06, gamma_h=0.6, gamma_l=0.9, theta=0.3,
                 alpha=0.6, delta=0.4, kappa_G=0.05, kappa_A=0.01,
                 P_max=1.5, M_Au=196.97, M_Ag=107.87):
        """
        Args:
            eta: Basic consumption need
            gamma_h: Storage fraction in abundance
            gamma_l: Consumption fraction in scarcity
            theta: Basculement threshold
            alpha: Gold savings fraction
            delta: Gold consumption fraction
            kappa_G: Gold minting cost
            kappa_A: Silver minting cost
            P_max: Maximum production
            M_Au: Gold molar mass (g/mol)
            M_Ag: Silver molar mass (g/mol)
        """
        self.eta = eta
        self.gamma_h = gamma_h
        self.gamma_l = gamma_l
        self.theta = theta
        self.alpha = alpha
        self.delta = delta
        self.kappa_G = kappa_G
        self.kappa_A = kappa_A
        self.P_max = P_max
        self.M_Au = M_Au
        self.M_Ag = M_Ag
        
        # Historical maximum stock
        self.S_max_hist = 0.0
    
    def production(self, t):
        """
        Yusufian production cycle: P(t) = P_max * sin²(2πt/7)
        """
        return self.P_max * np.sin(2 * np.pi * t / 7)**2
    
    def basculement(self, S, t):
        """
        Switching rule: gamma(t) = gamma_h if S > theta*S_max, else gamma_l
        """
        # Update historical maximum
        if S > self.S_max_hist:
            self.S_max_hist = S
        
        threshold = self.theta * self.S_max_hist
        
        if S > threshold:
            return self.gamma_h  # Abundance
        else:
            return self.gamma_l  # Scarcity
    
    def consumption(self, S, t):
        """
        Consumption: C(t) = eta * gamma(t)
        """
        gamma = self.basculement(S, t)
        return self.eta * gamma
    
    def minting_cost(self, savings, consumption, metal='gold'):
        """
        Minting cost: 10% of the metal flow.
        """
        if metal == 'gold':
            return 0.1 * (self.alpha * savings + self.delta * consumption)
        else:
            return 0.1 * ((1 - self.alpha) * savings + (1 - self.delta) * consumption)
    
    def system_dynamics(self, state, t):
        """
        Full system dynamics.
        
        state = [S, G, A]
        """
        S, G, A = state
        
        # Production
        P = self.production(t)
        
        # Consumption
        C = self.consumption(S, t)
        
        # Surplus
        surplus = max(0, P - C)
        
        # Minting costs
        mint_G = self.minting_cost(surplus, C, 'gold')
        mint_A = self.minting_cost(surplus, C, 'silver')
        
        # Stock derivatives
        dS = P - C
        dG = self.alpha * surplus - self.delta * C - self.kappa_G * mint_G
        dA = (1 - self.alpha) * surplus - (1 - self.delta) * C - self.kappa_A * mint_A
        
        # Ensure non-negative stocks
        if G + dG < 0:
            dG = -G
        if A + dA < 0:
            dA = -A
        
        return [dS, dG, dA]
    
    def simulate(self, S0=1.0, G0=100.0, A0=1000.0, T=100.0, dt=0.01):
        """
        Run simulation.
        
        Returns:
            Dictionary with results
        """
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize arrays
        S = np.zeros(n_steps)
        G = np.zeros(n_steps)
        A = np.zeros(n_steps)
        P = np.zeros(n_steps)
        C = np.zeros(n_steps)
        gamma = np.zeros(n_steps)
        P_G = np.zeros(n_steps)
        P_A = np.zeros(n_steps)
        R = np.zeros(n_steps)
        Lambda = np.zeros(n_steps)
        
        # Initial conditions
        S[0] = S0
        G[0] = G0
        A[0] = A0
        self.S_max_hist = S0
        
        for i in range(n_steps - 1):
            # Production
            P[i] = self.production(t[i])
            
            # Basculement
            gamma[i] = self.basculement(S[i], t[i])
            
            # Consumption
            C[i] = self.eta * gamma[i]
            
            # Surplus
            surplus = max(0, P[i] - C[i])
            
            # Minting costs
            mint_G = self.minting_cost(surplus, C[i], 'gold')
            mint_A = self.minting_cost(surplus, C[i], 'silver')
            
            # Stock updates (Euler)
            dS = P[i] - C[i]
            dG = self.alpha * surplus - self.delta * C[i] - self.kappa_G * mint_G
            dA = (1 - self.alpha) * surplus - (1 - self.delta) * C[i] - self.kappa_A * mint_A
            
            S[i+1] = max(0, S[i] + dS * dt)
            G[i+1] = max(0, G[i] + dG * dt)
            A[i+1] = max(0, A[i] + dA * dt)
            
            # Prices
            if G[i+1] > 0:
                P_G[i] = self.alpha * S[i] / G[i+1]
            else:
                P_G[i] = np.inf
            
            if A[i+1] > 0:
                P_A[i] = (1 - self.alpha) * S[i] / A[i+1]
            else:
                P_A[i] = np.inf
            
            # Exchange rate
            if A[i+1] > 0:
                R[i] = (G[i+1] / A[i+1]) * (self.M_Au / self.M_Ag)
            else:
                R[i] = np.inf
            
            # Lambda (zero debt)
            Lambda[i] = 0.0
        
        # Last values
        P[-1] = self.production(t[-1])
        gamma[-1] = self.basculement(S[-1], t[-1])
        C[-1] = self.eta * gamma[-1]
        
        if G[-1] > 0:
            P_G[-1] = self.alpha * S[-1] / G[-1]
        else:
            P_G[-1] = np.inf
        
        if A[-1] > 0:
            P_A[-1] = (1 - self.alpha) * S[-1] / A[-1]
            R[-1] = (G[-1] / A[-1]) * (self.M_Au / self.M_Ag)
        else:
            P_A[-1] = np.inf
            R[-1] = np.inf
        
        return {
            't': t,
            'S': S,
            'G': G,
            'A': A,
            'P': P,
            'C': C,
            'gamma': gamma,
            'P_G': P_G,
            'P_A': P_A,
            'R': R,
            'Lambda': Lambda
        }
    
    def compute_metrics(self, results):
        """
        Compute performance metrics.
        """
        G = results['G']
        A = results['A']
        R = results['R']
        
        # Filter infinite values
        R_finite = R[np.isfinite(R)]
        
        metrics = {
            'G_final': G[-1],
            'A_final': A[-1],
            'G_min': np.min(G),
            'G_max': np.max(G),
            'A_min': np.min(A),
            'A_max': np.max(A),
            'R_mean': np.mean(R_finite) if len(R_finite) > 0 else np.nan,
            'R_std': np.std(R_finite) if len(R_finite) > 0 else np.nan,
            'R_final': R[-1] if np.isfinite(R[-1]) else np.nan
        }
        
        return metrics

class DebtComparisonModel(NuqudModel):
    """
    Debt-based system for comparison with the nuqud system.
    """
    
    def __init__(self, D0=100.0, r=0.05, **kwargs):
        super().__init__(**kwargs)
        self.D0 = D0
        self.r = r
    
    def system_dynamics(self, state, t):
        """
        Debt-based system dynamics.
        state = [S, G, A, D]
        """
        S, G, A, D = state
        
        # Production
        P = self.production(t)
        
        # Consumption
        C = self.consumption(S, t)
        
        # Surplus
        surplus = max(0, P - C)
        
        # Minting costs
        mint_G = self.minting_cost(surplus, C, 'gold')
        mint_A = self.minting_cost(surplus, C, 'silver')
        
        # Debt dynamics
        dD = self.r * D
        
        # Stock derivatives (with debt service)
        debt_service = self.r * D
        dS = P - C - 0.8 * debt_service
        dG = self.alpha * surplus - self.delta * C - self.kappa_G * mint_G
        dA = (1 - self.alpha) * surplus - (1 - self.delta) * C - self.kappa_A * mint_A
        
        # Ensure non-negative
        if G + dG < 0:
            dG = -G
        if A + dA < 0:
            dA = -A
        
        return [dS, dG, dA, dD]
    
    def simulate(self, S0=1.0, G0=100.0, A0=1000.0, T=100.0, dt=0.01):
        """
        Run debt-based simulation.
        """
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize arrays
        S = np.zeros(n_steps)
        G = np.zeros(n_steps)
        A = np.zeros(n_steps)
        D = np.zeros(n_steps)
        P = np.zeros(n_steps)
        C = np.zeros(n_steps)
        gamma = np.zeros(n_steps)
        Lambda = np.zeros(n_steps)
        
        S[0] = S0
        G[0] = G0
        A[0] = A0
        D[0] = self.D0
        self.S_max_hist = S0
        
        for i in range(n_steps - 1):
            P[i] = self.production(t[i])
            gamma[i] = self.basculement(S[i], t[i])
            C[i] = self.eta * gamma[i]
            
            surplus = max(0, P[i] - C[i])
            debt_service = self.r * D[i]
            
            mint_G = self.minting_cost(surplus, C[i], 'gold')
            mint_A = self.minting_cost(surplus, C[i], 'silver')
            
            dS = P[i] - C[i] - 0.8 * debt_service
            dG = self.alpha * surplus - self.delta * C[i] - self.kappa_G * mint_G
            dA = (1 - self.alpha) * surplus - (1 - self.delta) * C[i] - self.kappa_A * mint_A
            dD = self.r * D[i]
            
            S[i+1] = max(0, S[i] + dS * dt)
            G[i+1] = max(0, G[i] + dG * dt)
            A[i+1] = max(0, A[i] + dA * dt)
            D[i+1] = D[i] + dD * dt
            
            # Lambda
            if S[i] > 0:
                Lambda[i] = (D[i] * self.r) / (S[i] + 1e-10)
            else:
                Lambda[i] = np.inf
        
        P[-1] = self.production(t[-1])
        gamma[-1] = self.basculement(S[-1], t[-1])
        C[-1] = self.eta * gamma[-1]
        
        return {
            't': t,
            'S': S,
            'G': G,
            'A': A,
            'D': D,
            'P': P,
            'C': C,
            'gamma': gamma,
            'Lambda': Lambda
        }
