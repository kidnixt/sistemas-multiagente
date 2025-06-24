import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, Tuple

def plot_strategy_evolution(strategy_history: Dict[str, list], iteration_history: list, 
                            info_set_keys: Optional[list] = None, 
                            action_names: Optional[Dict[int, str]] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot the evolution of strategies over training iterations.
    """
    if not iteration_history:
        print("No strategy history available.")
        return
    
    if action_names is None:
        action_names = {0: 'Pass', 1: 'Bet'}
    
    if info_set_keys is None:
        info_set_keys = list(set(key.split('_action_')[0] for key in strategy_history.keys()))
        info_set_keys = sorted(info_set_keys)
    
    n_plots = len(info_set_keys)
    if n_plots == 0:
        print("No information sets found to plot.")
        return
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten() if n_plots > 1 else [axes]
    
    for i, info_set_key in enumerate(info_set_keys):
        ax = axes_flat[i]
        max_actions = max(int(key.split('_action_')[1]) + 1 for key in strategy_history.keys() if key.startswith(f"{info_set_key}_action_"))
        
        for action_idx in range(max_actions):
            strategy_key = f"{info_set_key}_action_{action_idx}"
            if strategy_key in strategy_history:
                x_data = iteration_history[:len(strategy_history[strategy_key])]
                y_data = strategy_history[strategy_key]
                action_name = action_names.get(action_idx, f"Action {action_idx}")
                ax.plot(x_data, y_data, label=action_name, linewidth=2)
        
        ax.set_title(f"Strategy Evolution: {info_set_key}")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Action Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_convergence_analysis(strategy_history: Dict[str, list], iteration_history: list, 
                               info_set_key: str, 
                               action_names: Optional[Dict[int, str]] = None,
                               theoretical_values: Optional[Dict[int, float]] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot convergence analysis for a specific information set.
    """
    if action_names is None:
        action_names = {0: 'Pass', 1: 'Bet'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    available_actions = sorted(int(key.split('_action_')[1]) for key in strategy_history.keys() if key.startswith(f"{info_set_key}_action_"))
    
    for action_idx in available_actions:
        strategy_key = f"{info_set_key}_action_{action_idx}"
        if strategy_key in strategy_history:
            x_data = iteration_history[:len(strategy_history[strategy_key])]
            y_data = strategy_history[strategy_key]
            action_name = action_names.get(action_idx, f"Action {action_idx}")
            ax1.plot(x_data, y_data, label=f"Learned {action_name}", linewidth=2)
            
            if theoretical_values and action_idx in theoretical_values:
                ax1.axhline(y=theoretical_values[action_idx], color='red', linestyle='--', alpha=0.7, label=f"Theoretical {action_name}")
    
    ax1.set_title(f"Strategy Convergence: {info_set_key}")
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Action Probability")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    if theoretical_values:
        for action_idx in available_actions:
            strategy_key = f"{info_set_key}_action_{action_idx}"
            if strategy_key in strategy_history and action_idx in theoretical_values:
                errors = [abs(prob - theoretical_values[action_idx]) for prob in strategy_history[strategy_key]]
                x_data = iteration_history[:len(errors)]
                action_name = action_names.get(action_idx, f"Action {action_idx}")
                ax2.plot(x_data, errors, label=f"Error {action_name}", linewidth=2)
        
        ax2.set_title("Convergence Error")
        ax2.set_xlabel("Training Iteration")
        ax2.set_ylabel("Absolute Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No theoretical values\nprovided', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title("Theoretical Comparison")
    
    plt.tight_layout()
    plt.show()

def get_strategy_dataframe(strategy_history: Dict[str, list], iteration_history: list) -> pd.DataFrame:
    """
    Get strategy evolution as a pandas DataFrame for easier analysis.
    """
    if not iteration_history:
        return pd.DataFrame()
    
    data = {'iteration': iteration_history}
    for strategy_key, values in strategy_history.items():
        data[strategy_key] = values[:len(iteration_history)]
    
    return pd.DataFrame(data)