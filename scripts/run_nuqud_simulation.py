#!/usr/bin/env python3
"""
Run nuqud bimetallic system simulation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from src.nuqud_model import NuqudModel, DebtComparisonModel
from src.minting_optimization import MintingOptimizer
from src.visualization import FigureGenerator

def run_nuqud_simulation():
    """
    Run bimetallic system simulation.
    """
    print("=" * 60)
    print("Running Nuqud Bimetallic System Simulation...")
    print("=" * 60)
    
    # Initialize model
    model = NuqudModel(
        eta=0.06,
        gamma_h=0.6,
        gamma_l=0.9,
        theta=0.3,
        alpha=0.6,
        delta=0.4,
        kappa_G=0.05,
        kappa_A=0.01,
        P_max=1.5
    )
    
    # Run simulation
    results = model.simulate(
        S0=1.0,
        G0=100.0,
        A0=1000.0,
        T=100.0,
        dt=0.01
    )
    
    # Compute metrics
    metrics = model.compute_metrics(results)
    
    print("\nNuqud System Results:")
    print(f"  Final gold: {metrics['G_final']:.2f} oz")
    print(f"  Final silver: {metrics['A_final']:.2f} oz")
    print(f"  Gold range: [{metrics['G_min']:.2f}, {metrics['G_max']:.2f}]")
    print(f"  Silver range: [{metrics['A_min']:.2f}, {metrics['A_max']:.2f}]")
    print(f"  Exchange rate mean: {metrics['R_mean']:.3f}")
    print(f"  Exchange rate std: {metrics['R_std']:.3f}")
    print(f"  Lambda: {results['Lambda'][-1]:.4f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/results_nuqud.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_debt_simulation():
    """
    Run debt-based comparison simulation.
    """
    print("\n" + "=" * 60)
    print("Running Debt-Based Comparison Simulation...")
    print("=" * 60)
    
    model = DebtComparisonModel(
        D0=100.0,
        r=0.05,
        eta=0.06,
        gamma_h=0.6,
        gamma_l=0.9,
        theta=0.3,
        alpha=0.6,
        delta=0.4,
        kappa_G=0.05,
        kappa_A=0.01,
        P_max=1.5
    )
    
    results = model.simulate(
        S0=1.0,
        G0=100.0,
        A0=1000.0,
        T=100.0,
        dt=0.01
    )
    
    print("\nDebt-Based System Results:")
    print(f"  Final stock: {results['S'][-1]:.4f}")
    print(f"  Final debt: {results['D'][-1]:.2f}")
    print(f"  Final Lambda: {results['Lambda'][-1]:.4f}")
    
    # Save results
    with open('data/results_debt.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def run_minting_optimization():
    """
    Run minting optimization.
    """
    print("\n" + "=" * 60)
    print("Running Minting Optimization...")
    print("=" * 60)
    
    model = NuqudModel(
        eta=0.06,
        gamma_h=0.6,
        gamma_l=0.9,
        theta=0.3,
        alpha=0.6,
        delta=0.4,
        kappa_G=0.05,
        kappa_A=0.01,
        P_max=1.5
    )
    
    optimizer = MintingOptimizer(model)
    optimal_params, result = optimizer.optimize()
    optimal_alpha, optimal_delta = optimal_params
    
    print(f"\nMinting Optimization Results:")
    print(f"  Optimal alpha: {optimal_alpha:.3f}")
    print(f"  Optimal delta: {optimal_delta:.3f}")
    print(f"  Optimal ratio (silver/gold): {(1-optimal_alpha)/optimal_alpha:.3f}")
    print(f"  Optimization success: {result.success}")
    print(f"  Final objective: {result.fun:.2f}")
    
    # Save results
    with open('data/minting_results.pkl', 'wb') as f:
        pickle.dump({
            'optimal_alpha': optimal_alpha,
            'optimal_delta': optimal_delta,
            'optimization_result': result
        }, f)
    
    return optimizer, optimal_alpha, optimal_delta

def main():
    """
    Main execution function.
    """
    # Run simulations
    results_nuqud = run_nuqud_simulation()
    results_debt = run_debt_simulation()
    optimizer, optimal_alpha, optimal_delta = run_minting_optimization()
    
    print("\n" + "=" * 60)
    print("All simulations complete. Results saved to data/ directory.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_nuqud, results_debt, optimizer, optimal_alpha, optimal_delta)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
