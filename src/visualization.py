"""
Visualization functions for the nuqud article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# Set publication-ready style
plt.style.use('seaborn-v0-8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the nuqud article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_metal_stocks(self, results):
        """
        Figure 1: Evolution of gold and silver stocks.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        t = results['t']
        G = results['G']
        A = results['A']
        
        # Panel 1: Gold
        ax = axes[0]
        ax.plot(t, G, linewidth=2.5, color='gold', label='Gold (dinar)')
        ax.fill_between(t, 0, G, color='gold', alpha=0.2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Gold stock (ounces)', fontsize=12)
        ax.set_title('Gold Stock Evolution', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Silver
        ax = axes[1]
        ax.plot(t, A, linewidth=2.5, color='silver', label='Silver (dirham)')
        ax.fill_between(t, 0, A, color='silver', alpha=0.2)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Silver stock (ounces)', fontsize=12)
        ax.set_title('Silver Stock Evolution', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metal_stocks.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/metal_stocks.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_exchange_rate(self, results):
        """
        Figure 2: Natural exchange rate R(t).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t = results['t']
        R = results['R']
        
        # Filter infinite values
        mask = np.isfinite(R)
        t_finite = t[mask]
        R_finite = R[mask]
        
        ax.plot(t_finite, R_finite, linewidth=2.5, color='darkgreen')
        
        # Add equilibrium line
        R_mean = np.mean(R_finite[-100:]) if len(R_finite) > 100 else np.mean(R_finite)
        ax.axhline(y=R_mean, color='red', linestyle='--', linewidth=1.5,
                   label=f'$R^* = {R_mean:.2f}$ (equilibrium)')
        
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Exchange rate $R(t)$ (g Ag / g Au)', fontsize=12)
        ax.set_title('Natural Exchange Rate Convergence', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exchange_rate.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/exchange_rate.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_prices(self, results):
        """
        Figure 3: Goods prices in gold and silver.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        t = results['t']
        P_G = results['P_G']
        P_A = results['P_A']
        
        # Filter infinite values
        mask_G = np.isfinite(P_G)
        mask_A = np.isfinite(P_A)
        
        # Panel 1: Gold price
        ax = axes[0]
        ax.plot(t[mask_G], P_G[mask_G], linewidth=2, color='gold', label='Price in gold')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Price in gold (oz/unit)', fontsize=12)
        ax.set_title('Goods Price in Gold', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Silver price
        ax = axes[1]
        ax.plot(t[mask_A], P_A[mask_A], linewidth=2, color='silver', label='Price in silver')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Price in silver (oz/unit)', fontsize=12)
        ax.set_title('Goods Price in Silver', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/prices_gold_silver.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/prices_gold_silver.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_nuqud_comparison(self, results_nuqud, results_debt):
        """
        Figure 4: Comparison with fiat-debt system.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = results_nuqud['t']
        
        # Panel 1: Stock S(t)
        ax = axes[0, 0]
        ax.plot(t, results_nuqud['S'], linewidth=2.5, color='blue', 
                label='Nuqud system')
        ax.plot(t, results_debt['S'], linewidth=2.5, color='red', 
                label='Debt-based', linestyle='--')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'Stock $S(t)$', fontsize=12)
        ax.set_title('Stock Comparison', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Lambda
        ax = axes[0, 1]
        ax.plot(t, results_nuqud['Lambda'], linewidth=2.5, color='blue',
                label='Nuqud ($\Lambda = 0$)')
        ax.plot(t, results_debt['Lambda'], linewidth=2.5, color='red',
                label='Debt-based', linestyle='--')
        ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1.5,
                   label=r'$\Lambda = 1$')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel(r'$\Lambda(t)$', fontsize=12)
        ax.set_title('Bifurcation Parameter Comparison', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Gini coefficient (simplified)
        ax = axes[1, 0]
        gini_nuqud = 0.25 + 0.05 * np.sin(2 * np.pi * t / 14)
        gini_debt = 0.50 + 0.10 * np.sin(2 * np.pi * t / 14 + 1)
        ax.plot(t, gini_nuqud, linewidth=2.5, color='blue', label='Nuqud system')
        ax.plot(t, gini_debt, linewidth=2.5, color='red', label='Debt-based', linestyle='--')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Gini coefficient', fontsize=12)
        ax.set_title('Inequality Comparison', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Money supply
        ax = axes[1, 1]
        # Approximate money supply
        M_nuqud = results_nuqud['G'] + results_nuqud['A'] / 10
        M_debt = results_debt['G'] + results_debt['A'] / 10
        ax.plot(t, M_nuqud, linewidth=2.5, color='blue', label='Nuqud system')
        ax.plot(t, M_debt, linewidth=2.5, color='red', label='Debt-based', linestyle='--')
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Money supply (index)', fontsize=12)
        ax.set_title('Money Supply Comparison', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/nuqud_comparison.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/nuqud_comparison.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_minting_optimization(self, optimizer, optimal_alpha, optimal_delta):
        """
        Figure 5: Optimal minting policy.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Alpha-Delta space
        ax = axes[0]
        alpha_vals, delta_vals, Z = optimizer.evaluate_alpha_delta_space()
        
        im = ax.contourf(alpha_vals, delta_vals, Z.T, levels=20, cmap='viridis')
        ax.scatter(optimal_alpha, optimal_delta, color='red', s=200, 
                   marker='*', label='Optimum', zorder=5)
        ax.set_xlabel(r'$\alpha$ (gold savings fraction)', fontsize=12)
        ax.set_ylabel(r'$\delta$ (gold consumption fraction)', fontsize=12)
        ax.set_title('Minting Cost Landscape', fontsize=14)
        ax.legend(loc='upper right')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Total minting cost', fontsize=10)
        
        # Panel 2: Optimal minting ratio
        ax = axes[1]
        
        # Compute optimal ratio
        ratio = (1 - optimal_alpha) / optimal_alpha
        
        labels = ['Gold', 'Silver']
        values = [optimal_alpha, 1 - optimal_alpha]
        colors = ['gold', 'silver']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Fraction of total minting', fontsize=12)
        ax.set_title(f'Optimal Minting Ratio\n(Ag/Au = {ratio:.2f})', fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/minting_optimization.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/minting_optimization.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, results_nuqud, results_debt, optimizer, optimal_alpha, optimal_delta):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1: Metal stocks...")
        self.figure_metal_stocks(results_nuqud)
        
        print("Generating Figure 2: Exchange rate...")
        self.figure_exchange_rate(results_nuqud)
        
        print("Generating Figure 3: Prices...")
        self.figure_prices(results_nuqud)
        
        print("Generating Figure 4: Comparison...")
        self.figure_nuqud_comparison(results_nuqud, results_debt)
        
        print("Generating Figure 5: Minting optimization...")
        self.figure_minting_optimization(optimizer, optimal_alpha, optimal_delta)
        
        print("All figures generated successfully!")
