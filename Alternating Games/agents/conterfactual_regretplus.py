from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional
import copy
import matplotlib.pyplot as plt
import pandas as pd

class InfoSetPlus:
    def __init__(self, num_actions: int):
        self.regrets = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.ones(num_actions) / num_actions
        self.cumulative_regrets = np.zeros(num_actions)  # For CFR+
        
    def get_strategy(self, iteration: int = 0) -> np.ndarray:
        """Get current strategy using regret matching+"""
        # CFR+: Use cumulative regrets instead of immediate regrets
        positive_regrets = np.maximum(self.cumulative_regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 1e-15:
            self.current_strategy = positive_regrets / regret_sum
        else:
            self.current_strategy = np.ones(len(self.regrets)) / len(self.regrets)
        
        return self.current_strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        norm_sum = np.sum(self.strategy_sum)
        if norm_sum > 1e-15:
            return self.strategy_sum / norm_sum
        else:
            return np.ones(len(self.strategy_sum)) / len(self.strategy_sum)
    
    def update_regrets_plus(self, regret_updates: np.ndarray, iteration: int):
        """CFR+ regret update: only accumulate positive regrets"""
        # Update immediate regrets
        self.regrets += regret_updates
        
        # CFR+: Update cumulative regrets with positive values only
        # Reset negative cumulative regrets to 0
        self.cumulative_regrets = np.maximum(self.cumulative_regrets + regret_updates, 0)
    
    def update_strategy_sum_plus(self, strategy_updates: np.ndarray, iteration: int):
        """CFR+ strategy sum update with discount factor"""
        # In CFR+, we can use linear weighting or other schemes
        # For simplicity, using standard accumulation here
        self.strategy_sum += strategy_updates

class CounterFactualRegretPlus(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, seed: Optional[int] = None, 
                 track_frequency: int = 1, alternating_updates: bool = True):
        super().__init__(game, agent)
        self.info_sets: Dict[str, InfoSetPlus] = {}
        self.node_dict = {}  # For compatibility with notebook expectations
        
        # CFR+ specific parameters
        self.alternating_updates = alternating_updates
        
        # Strategy tracking for plotting
        self.strategy_history = defaultdict(list)
        self.iteration_history = []
        self.track_frequency = track_frequency
        
        if seed is not None:
            np.random.seed(seed)
            
    def get_info_set_key(self, game_state: AlternatingGame) -> str:
        """Create a unique key for the information set based on game state"""
        obs = game_state.observe(game_state.agent_selection)
        if isinstance(obs, dict):
            return str(sorted(obs.items()))
        return str(obs)
    
    def get_or_create_info_set(self, key: str, num_actions: int) -> InfoSetPlus:
        """Get existing info set or create new one"""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetPlus(num_actions)
        return self.info_sets[key]
    
    def cfr_plus(self, game_state: AlternatingGame, player: AgentID, reach_prob: Dict[AgentID, float], 
                 chance_reach: float = 1.0, update_player: Optional[AgentID] = None, 
                 iteration: int = 0) -> float:
        """
        CFR+ algorithm implementation
        """
        # Terminal node
        if game_state.terminated():
            reward = game_state.reward(update_player)
            return reward if reward is not None else 0.0
        
        # Get current player
        current_player = game_state.agent_selection
        if update_player is None:
            update_player = current_player
            
        # Get available actions
        actions = game_state.available_actions()
        num_actions = len(actions)
        
        if num_actions == 0:
            return 0.0
        
        # Get information set
        info_set_key = self.get_info_set_key(game_state)
        info_set = self.get_or_create_info_set(info_set_key, num_actions)
        
        # Get current strategy using CFR+
        strategy = info_set.get_strategy(iteration)
        
        # Compute utilities for each action
        action_utilities = np.zeros(num_actions)
        node_utility = 0.0
        
        for i, action in enumerate(actions):
            # Create new game state
            new_game_state = copy.deepcopy(game_state)  # Use deepcopy for safety
            new_game_state.step(action)
            
            # Update reach probabilities
            new_reach_prob = reach_prob.copy()
            new_reach_prob[current_player] *= strategy[i]
            
            # Recursive call
            action_utilities[i] = self.cfr_plus(
                new_game_state, 
                new_game_state.agent_selection if not new_game_state.terminated() else current_player,
                new_reach_prob, 
                chance_reach, 
                update_player,
                iteration
            )
            
            node_utility += strategy[i] * action_utilities[i]
        
        # CFR+ Updates
        if current_player == update_player:
            # Calculate counterfactual reach
            cfr_reach = chance_reach
            for p, prob in reach_prob.items():
                if p != current_player:
                    cfr_reach *= prob
            
            # CFR+ alternating updates
            update_regrets = True
            update_strategy = True
            
            if self.alternating_updates:
                # Alternate between regret and strategy updates
                if iteration % 2 == 0:
                    update_strategy = False  # Update only regrets on even iterations
                else:
                    update_regrets = False   # Update only strategy on odd iterations
            
            # Update regrets using CFR+
            if update_regrets and abs(cfr_reach) > 1e-10:
                regret_updates = np.zeros(num_actions)
                for i in range(num_actions):
                    regret = action_utilities[i] - node_utility
                    regret_updates[i] = cfr_reach * regret
                
                # Use CFR+ regret update
                info_set.update_regrets_plus(regret_updates, iteration)
            
            # Update strategy sum using CFR+
            if update_strategy:
                my_reach = reach_prob.get(current_player, 1.0)
                if abs(my_reach) > 1e-10:
                    strategy_updates = my_reach * strategy
                    info_set.update_strategy_sum_plus(strategy_updates, iteration)
        
        return node_utility
    
    def train(self, iterations: int = 10000, track_strategies: bool = True, 
              verbose: bool = False) -> None:
        """Train the agent using CFR+ with optional strategy tracking"""
        
        for i in range(iterations):
            # Train for each player
            for player in self.game.agents:
                # Ensure clean game state
                self.game.reset()
                
                # Verify game is in initial state
                if not hasattr(self.game, 'agent_selection') or self.game.agent_selection is None:
                    continue
                    
                reach_prob = {p: 1.0 for p in self.game.agents}
                
                try:
                    utility = self.cfr_plus(
                        self.game, 
                        self.game.agent_selection, 
                        reach_prob, 
                        1.0, 
                        player, 
                        i
                    )
                    
                    if verbose and i % 1000 == 0:
                        print(f"Iteration {i}, Player {player}, Utility: {utility:.6f}")
                        
                except Exception as e:
                    if verbose:
                        print(f"Error during CFR+ iteration {i} for player {player}: {e}")
                    continue
            
            # Track strategies at specified intervals
            if track_strategies and i % self.track_frequency == 0:
                self._track_current_strategies(i)
        
        # Create node_dict for compatibility with existing notebooks
        self._create_node_dict()
        
        if verbose:
            print(f"CFR+ training completed after {iterations} iterations")
    
    def _create_node_dict(self) -> None:
        """Create node_dict for compatibility with existing code"""
        self.node_dict = {}
        for key, info_set in self.info_sets.items():
            class PolicyWrapper:
                def __init__(self, info_set):
                    self.info_set = info_set
                
                def policy(self):
                    return self.info_set.get_average_strategy()
            
            self.node_dict[key] = PolicyWrapper(info_set)
    
    def action(self) -> int:
        """Choose action based on current strategy"""
        info_set_key = self.get_info_set_key(self.game)
        
        if info_set_key in self.info_sets:
            actions = self.game.available_actions()
            strategy = self.info_sets[info_set_key].get_average_strategy()
            
            if len(strategy) != len(actions):
                return np.random.choice(actions)
            
            return np.random.choice(actions, p=strategy)
        else:
            return np.random.choice(self.game.available_actions())
    
    def policy(self) -> Dict[str, np.ndarray]:
        """Return the learned policies for all information sets"""
        policies = {}
        for key, info_set in self.info_sets.items():
            policies[key] = info_set.get_average_strategy()
        return policies
    
    def get_strategy(self, info_set_key: str) -> np.ndarray:
        """Get strategy for a specific information set"""
        if info_set_key in self.info_sets:
            return self.info_sets[info_set_key].get_average_strategy()
        else:
            actions = self.game.available_actions()
            return np.ones(len(actions)) / len(actions)
    
    def _track_current_strategies(self, iteration: int) -> None:
        """Track current average strategies for plotting"""
        self.iteration_history.append(iteration)
        
        current_info_sets = set(self.info_sets.keys())
        
        for key in current_info_sets:
            info_set = self.info_sets[key]
            avg_strategy = info_set.get_average_strategy()
            
            for action_idx, prob in enumerate(avg_strategy):
                strategy_key = f"{key}_action_{action_idx}"
                
                if strategy_key not in self.strategy_history:
                    self.strategy_history[strategy_key] = [prob] * len(self.iteration_history)
                else:
                    self.strategy_history[strategy_key].append(prob)
        
        # Ensure all existing strategy keys have the same length
        for strategy_key in list(self.strategy_history.keys()):
            while len(self.strategy_history[strategy_key]) < len(self.iteration_history):
                last_value = self.strategy_history[strategy_key][-1] if self.strategy_history[strategy_key] else 0.5
                self.strategy_history[strategy_key].append(last_value)

    def compare_regrets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compare immediate regrets vs cumulative regrets for analysis"""
        comparison = {}
        for key, info_set in self.info_sets.items():
            comparison[key] = {
                'immediate_regrets': info_set.regrets.copy(),
                'cumulative_regrets': info_set.cumulative_regrets.copy(),
                'strategy_sum': info_set.strategy_sum.copy()
            }
        return comparison

    # Include all the plotting methods from the original class
    def plot_strategy_evolution(self, info_set_keys: Optional[list] = None, 
                            action_names: Optional[Dict[int, str]] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot the evolution of strategies over training iterations"""
        if not self.iteration_history:
            print("No strategy history available. Make sure to train with track_strategies=True")
            return
        
        if action_names is None:
            action_names = {0: 'Pass', 1: 'Bet'}
        
        if info_set_keys is None:
            info_set_keys = list(set(key.split('_action_')[0] for key in self.strategy_history.keys()))
            info_set_keys = sorted(info_set_keys)
        
        n_plots = len(info_set_keys)
        if n_plots == 0:
            print("No information sets found to plot")
            return
            
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if n_plots > 1 else axes
        
        for i, info_set_key in enumerate(info_set_keys):
            ax = axes_flat[i] if n_plots > 1 else axes_flat[0]
            
            max_actions = 0
            for strategy_key in self.strategy_history.keys():
                if strategy_key.startswith(f"{info_set_key}_action_"):
                    action_idx = int(strategy_key.split('_action_')[1])
                    max_actions = max(max_actions, action_idx + 1)
            
            for action_idx in range(max_actions):
                strategy_key = f"{info_set_key}_action_{action_idx}"
                if strategy_key in self.strategy_history:
                    data_length = min(len(self.iteration_history), len(self.strategy_history[strategy_key]))
                    x_data = self.iteration_history[:data_length]
                    y_data = self.strategy_history[strategy_key][:data_length]
                    
                    action_name = action_names.get(action_idx, f"Action {action_idx}")
                    ax.plot(x_data, y_data, label=action_name, linewidth=2)
            
            ax.set_title(f"CFR+ Strategy Evolution: {info_set_key}")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Action Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_regret_comparison(self, info_set_key: str, figsize: Tuple[int, int] = (12, 5)) -> None:
        """Plot comparison between immediate and cumulative regrets"""
        if info_set_key not in self.info_sets:
            print(f"Information set '{info_set_key}' not found")
            return
        
        info_set = self.info_sets[info_set_key]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        actions = range(len(info_set.regrets))
        
        # Plot immediate regrets
        ax1.bar(actions, info_set.regrets, alpha=0.7, color='blue')
        ax1.set_title(f"Immediate Regrets: {info_set_key}")
        ax1.set_xlabel("Action")
        ax1.set_ylabel("Regret")
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative regrets (CFR+)
        ax2.bar(actions, info_set.cumulative_regrets, alpha=0.7, color='red')
        ax2.set_title(f"Cumulative Regrets (CFR+): {info_set_key}")
        ax2.set_xlabel("Action")
        ax2.set_ylabel("Cumulative Regret")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_convergence_analysis(self, info_set_key: str, 
                                action_names: Optional[Dict[int, str]] = None,
                                theoretical_values: Optional[Dict[int, float]] = None,
                                figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot convergence analysis for a specific information set
        """
        if action_names is None:
            action_names = {0: 'Pass', 1: 'Bet'}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Find available actions for this info set
        available_actions = []
        for strategy_key in self.strategy_history.keys():
            if strategy_key.startswith(f"{info_set_key}_action_"):
                action_idx = int(strategy_key.split('_action_')[1])
                available_actions.append(action_idx)
        
        available_actions = sorted(available_actions)
        
        # Plot 1: Strategy evolution
        for action_idx in available_actions:
            strategy_key = f"{info_set_key}_action_{action_idx}"
            if strategy_key in self.strategy_history:
                # Ensure data length matches iteration history
                data_length = min(len(self.iteration_history), len(self.strategy_history[strategy_key]))
                x_data = self.iteration_history[:data_length]
                y_data = self.strategy_history[strategy_key][:data_length]
                
                action_name = action_names.get(action_idx, f"Action {action_idx}")
                ax1.plot(x_data, y_data, label=f"Learned {action_name}", linewidth=2)
                
                # Add theoretical line if provided
                if theoretical_values and action_idx in theoretical_values:
                    ax1.axhline(y=theoretical_values[action_idx], 
                            color='red', linestyle='--', alpha=0.7,
                            label=f"Theoretical {action_name}")
        
        ax1.set_title(f"Strategy Convergence: {info_set_key}")
        ax1.set_xlabel("Training Iteration")
        ax1.set_ylabel("Action Probability")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Convergence error (if theoretical values provided)
        if theoretical_values:
            for action_idx in available_actions:
                strategy_key = f"{info_set_key}_action_{action_idx}"
                if strategy_key in self.strategy_history and action_idx in theoretical_values:
                    data_length = min(len(self.iteration_history), len(self.strategy_history[strategy_key]))
                    errors = [abs(prob - theoretical_values[action_idx]) 
                            for prob in self.strategy_history[strategy_key][:data_length]]
                    x_data = self.iteration_history[:data_length]
                    
                    action_name = action_names.get(action_idx, f"Action {action_idx}")
                    ax2.plot(x_data, errors, label=f"Error {action_name}", linewidth=2)
            
            ax2.set_title("Convergence Error")
            ax2.set_xlabel("Training Iteration")
            ax2.set_ylabel("Absolute Error")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No theoretical values\nprovided', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title("Theoretical Comparison")
        
        plt.tight_layout()
        plt.show()