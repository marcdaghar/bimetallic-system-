#!/usr/bin/env python3
"""
Generate all figures from saved results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from src.nuqud_model import NuqudModel
from src.minting_optimization import MintingOptimizer
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate figures.
    """
    print("Loading saved results...")
    
    with open('data/results_nuqud.pkl', 'rb') as f:
        results_nuqud = pickle.load(f)
    
    with open('data/results_debt.pkl', 'rb') as f:
        results_debt = pickle.load(f)
    
    with open('data/minting_results.pkl', 'rb') as f:
        minting_results = pickle.load(f)
    
    optimal_alpha = minting_results['optimal_alpha']
    optimal_delta = minting_results['optimal_delta']
    
    # Recreate optimizer for visualization
    model = NuqudModel()
    optimizer = MintingOptimizer(model)
    
    print("Generating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(results_nuqud, results_debt, optimizer, optimal_alpha, optimal_delta)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
