import numpy as np
from base.agent import Agent, AgentID
from base.game import AlternatingGame
from collections import defaultdict
from typing import Dict, Optional, Tuple
import copy
import matplotlib.pyplot as plt
import pandas as pd

class InfoSetSampling:
    def __init__(self, num_actions: int):
        self.regrets = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.ones(num_actions) / num_actions

    def get_strategy(self) -> np.ndarray:
        pos_regrets = np.maximum(self.regrets, 0)
        smoothed_regrets = pos_regrets + 1e-3
        regret_sum = smoothed_regrets.sum()
        if regret_sum > 1e-15:
            self.current_strategy = smoothed_regrets / regret_sum
        else:
            self.current_strategy = np.ones_like(self.regrets) / len(self.regrets)
        return self.current_strategy

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 1e-15:
            return self.strategy_sum / total
        else:
            return np.ones_like(self.strategy_sum) / len(self.strategy_sum)

    def update_regrets(self, regret_update: np.ndarray):
        self.regrets += regret_update

    def update_strategy_sum(self, strategy: np.ndarray, weight: float = 1.0):
        self.strategy_sum += weight * strategy


class CounterFactualRegretSampling(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, seed: Optional[int] = None, track_frequency: int = 100) -> None:
        super().__init__(game, agent)
        self.info_sets: Dict[str, InfoSetSampling] = {}
        self.node_dict = {}
        if seed is not None:
            np.random.seed(seed)

        # Strategy tracking for plotting
        self.strategy_history = defaultdict(list)
        self.iteration_history = []
        self.track_frequency = track_frequency

    def get_info_set_key(self, game_state: AlternatingGame) -> str:
        obs = game_state.observe(game_state.agent_selection)
        return str(sorted(obs.items())) if isinstance(obs, dict) else str(obs)

    def get_or_create_info_set(self, key: str, num_actions: int) -> InfoSetSampling:
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetSampling(num_actions)
        return self.info_sets[key]

    def external_sampling_cfr(self, game_state: AlternatingGame, player: AgentID,
                               pi: float, sigma: float) -> float:
        if game_state.terminated():
            reward = game_state.reward(player)
            return reward if reward is not None else 0.0

        current_player = game_state.agent_selection
        actions = game_state.available_actions()
        num_actions = len(actions)

        info_set_key = self.get_info_set_key(game_state)
        info_set = self.get_or_create_info_set(info_set_key, num_actions)
        strategy = info_set.get_strategy()

        if current_player == player:
            # Traverse all actions for own player
            util = np.zeros(num_actions)
            node_utility = 0.0
            for i, action in enumerate(actions):
                new_state = copy.deepcopy(game_state)
                new_state.step(action)
                util[i] = self.external_sampling_cfr(new_state, player, pi * strategy[i], sigma)
                node_utility += strategy[i] * util[i]

            regrets = (util - node_utility) * sigma
            info_set.update_regrets(regrets)
            info_set.update_strategy_sum(strategy, pi)
            return node_utility
        else:
            # Sample one action from opponent's strategy
            action_idx = np.random.choice(len(actions), p=strategy)
            sampled_action = actions[action_idx]
            new_state = copy.deepcopy(game_state)
            new_state.step(sampled_action)
            return self.external_sampling_cfr(new_state, player, pi, sigma * strategy[action_idx])

    def train(self, iterations: int = 10000, verbose: bool = False, track_strategies: bool = True) -> None:
        for i in range(iterations):
            for player in self.game.agents:
                self.game.reset()
                self.external_sampling_cfr(self.game, player, pi=1.0, sigma=1.0)

                if verbose and i % 1000 == 0:
                    print(f"Iteration {i}, Player {player}")

            if track_strategies and i % self.track_frequency == 0:
                self._track_current_strategies(i)

        self._create_node_dict()

    def _create_node_dict(self) -> None:
        self.node_dict = {}
        for key, info_set in self.info_sets.items():
            class PolicyWrapper:
                def __init__(self, info_set):
                    self.info_set = info_set
                def policy(self):
                    return self.info_set.get_average_strategy()
            self.node_dict[key] = PolicyWrapper(info_set)

    def action(self) -> int:
        key = self.get_info_set_key(self.game)
        actions = self.game.available_actions()
        if key in self.info_sets:
            strategy = self.info_sets[key].get_average_strategy()
            return np.random.choice(actions, p=strategy)
        return np.random.choice(actions)

    def policy(self) -> Dict[str, np.ndarray]:
        return {k: v.get_average_strategy() for k, v in self.info_sets.items()}
    
    def _track_current_strategies(self, iteration: int) -> None:
        """Track current average strategies for plotting"""
        self.iteration_history.append(iteration)
        
        # Get all current info sets
        current_info_sets = set(self.info_sets.keys())
        
        for key in current_info_sets:
            info_set = self.info_sets[key]
            avg_strategy = info_set.get_average_strategy()
            
            # Store each action probability
            for action_idx, prob in enumerate(avg_strategy):
                strategy_key = f"{key}_action_{action_idx}"
                
                # Initialize list if this is the first time we see this strategy key
                if strategy_key not in self.strategy_history:
                    # Pad with current value for all previous iterations
                    self.strategy_history[strategy_key] = [prob] * len(self.iteration_history)
                else:
                    self.strategy_history[strategy_key].append(prob)
        
        # Ensure all existing strategy keys have the same length as iteration_history
        for strategy_key in list(self.strategy_history.keys()):
            while len(self.strategy_history[strategy_key]) < len(self.iteration_history):
                # If a strategy key is missing data, pad with the last known value
                last_value = self.strategy_history[strategy_key][-1] if self.strategy_history[strategy_key] else 0.5
                self.strategy_history[strategy_key].append(last_value)

    def plot_strategy_evolution(self, info_set_keys: Optional[list] = None, 
                            action_names: Optional[Dict[int, str]] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the evolution of strategies over training iterations
        """
        if not self.iteration_history:
            print("No strategy history available. Make sure to train with track_strategies=True")
            return
        
        # Default action names
        if action_names is None:
            action_names = {0: 'Pass', 1: 'Bet'}
        
        # Filter information sets to plot
        if info_set_keys is None:
            info_set_keys = list(set(key.split('_action_')[0] for key in self.strategy_history.keys()))
            info_set_keys = sorted(info_set_keys)  # Sort for consistent ordering
        
        # Create subplots
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
            
            # Find the maximum number of actions for this info set
            max_actions = 0
            for strategy_key in self.strategy_history.keys():
                if strategy_key.startswith(f"{info_set_key}_action_"):
                    action_idx = int(strategy_key.split('_action_')[1])
                    max_actions = max(max_actions, action_idx + 1)
            
            # Plot each action's probability over time
            for action_idx in range(max_actions):
                strategy_key = f"{info_set_key}_action_{action_idx}"
                if strategy_key in self.strategy_history:
                    # Ensure data length matches iteration history
                    data_length = min(len(self.iteration_history), len(self.strategy_history[strategy_key]))
                    x_data = self.iteration_history[:data_length]
                    y_data = self.strategy_history[strategy_key][:data_length]
                    
                    action_name = action_names.get(action_idx, f"Action {action_idx}")
                    ax.plot(x_data, y_data, label=action_name, linewidth=2)
            
            ax.set_title(f"Strategy Evolution: {info_set_key}")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Action Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
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

    def get_strategy_dataframe(self) -> pd.DataFrame:
        """Get strategy evolution as a pandas DataFrame for easier analysis"""
        if not self.iteration_history:
            return pd.DataFrame()
        
        data = {'iteration': self.iteration_history}
        
        for strategy_key, values in self.strategy_history.items():
            # Ensure data length matches iteration history
            data_length = min(len(self.iteration_history), len(values))
            data[strategy_key] = values[:data_length]
        
        return pd.DataFrame(data)
