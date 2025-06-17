from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional
import copy
import matplotlib.pyplot as plt
import pandas as pd

class InfoSet:
    def __init__(self, num_actions: int):
        self.regrets = np.random.normal(0, 1e-6, num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.ones(num_actions) / num_actions
        self.regret_floor = -300
        self.t = 0  # Time step counter for discounting
        
    def update_regrets_and_strategy(self, regret_updates: np.ndarray, strategy: np.ndarray, reach_prob: float):
        """Update regrets and strategy sum with optional discounting"""
        self.t += 1
        
        # Update regrets with floor
        self.regrets += regret_updates
        self.regrets = np.maximum(self.regrets, self.regret_floor)
        
        # Use linear averaging for strategy sum (prevents explosion)
        discount_factor = self.t / (self.t + 1) if self.t > 1000 else 1.0
        self.strategy_sum = self.strategy_sum * discount_factor + strategy * reach_prob
        
    def get_strategy(self) -> np.ndarray:
        """Get current strategy using regret matching with floor"""
        # Apply regret floor to prevent extreme negative values
        clamped_regrets = np.maximum(self.regrets, self.regret_floor)
        positive_regrets = np.maximum(clamped_regrets, 0)
        
        regret_sum = np.sum(positive_regrets)
        if regret_sum > 1e-15:
            self.current_strategy = positive_regrets / regret_sum
        else:
            self.current_strategy = np.ones(len(self.regrets)) / len(self.regrets)
        return self.current_strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        norm_sum = np.sum(self.strategy_sum)
        if norm_sum > 1e-15:  # Use small epsilon instead of 0
            return self.strategy_sum / norm_sum
        else:
            return np.ones(len(self.strategy_sum)) / len(self.strategy_sum)

class CounterFactualRegret(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, seed: Optional[int] = None, 
                 track_frequency: int = 1, exploration_bonus: float = 0.01):
        super().__init__(game, agent)
        self.info_sets: Dict[str, InfoSet] = {}
        self.node_dict = {}
        self.exploration_bonus = exploration_bonus
        
        # Strategy tracking for plotting
        self.strategy_history = defaultdict(list)
        self.iteration_history = []
        self.track_frequency = track_frequency
        
        if seed is not None:
            np.random.seed(seed)
            
    def get_info_set_key(self, game_state: AlternatingGame) -> str:
        """Create a unique key for the information set based on game state"""
        # This is game-specific and should be adapted based on your game
        # For now, using a simple observation-based key
        obs = game_state.observe(game_state.agent_selection)
        if isinstance(obs, dict):
            return str(sorted(obs.items()))
        return str(obs)
    
    def get_or_create_info_set(self, key: str, num_actions: int) -> InfoSet:
        """Get existing info set or create new one"""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet(num_actions)
        return self.info_sets[key]
    
    def cfr(self, game_state: AlternatingGame, player: AgentID, reach_prob: Dict[AgentID, float], 
            chance_reach: float = 1.0, update_player: Optional[AgentID] = None) -> float:
        """CFR with exploration bonus and numerical stability"""
        
        if game_state.terminated():
            reward = game_state.reward(update_player)
            return reward if reward is not None else 0.0
        
        current_player = game_state.agent_selection
        if update_player is None:
            update_player = current_player
            
        actions = game_state.available_actions()
        num_actions = len(actions)
        
        info_set_key = self.get_info_set_key(game_state)
        info_set = self.get_or_create_info_set(info_set_key, num_actions)
        
        # Get current strategy with exploration bonus
        base_strategy = info_set.get_strategy()
        strategy = (1 - self.exploration_bonus) * base_strategy + \
                  (self.exploration_bonus / num_actions)
        
        # Normalize to ensure it's a valid probability distribution
        strategy = strategy / np.sum(strategy)
        
        action_utilities = np.zeros(num_actions)
        node_utility = 0.0
        
        for i, action in enumerate(actions):
            new_game_state = game_state.clone()
            new_game_state.step(action)
            
            new_reach_prob = reach_prob.copy()
            new_reach_prob[current_player] *= strategy[i]
            
            action_utilities[i] = self.cfr(
                new_game_state, 
                new_game_state.agent_selection if not new_game_state.terminated() else current_player,
                new_reach_prob, 
                chance_reach, 
                update_player
            )
            
            node_utility += strategy[i] * action_utilities[i]
        
        # Update regrets and strategy sum if this is the player we're updating
        if current_player == update_player:
            cfr_reach = chance_reach
            for p, prob in reach_prob.items():
                if p != current_player:
                    cfr_reach *= prob
            
            # Calculate regret updates
            regret_updates = np.zeros(num_actions)
            for i in range(num_actions):
                regret_updates[i] = cfr_reach * (action_utilities[i] - node_utility)
            
            # Update using the info set's method
            my_reach = reach_prob.get(current_player, 1.0)
            if hasattr(info_set, 'update_regrets_and_strategy'):
                info_set.update_regrets_and_strategy(regret_updates, base_strategy, my_reach)
            else:
                # Fallback to original method with clamping
                info_set.regrets += regret_updates
                info_set.regrets = np.maximum(info_set.regrets, -300)  # Regret floor
                info_set.strategy_sum += my_reach * base_strategy
        
        return node_utility
    
    def train(self, iterations: int = 10000, track_strategies: bool = True) -> None:
        """Train the agent using CFR with optional strategy tracking"""            
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
                    self.cfr(self.game, self.game.agent_selection, reach_prob, 1.0, player)
                except Exception as e:
                    print(f"Error during CFR iteration {i} for player {player}: {e}")
                    continue
            
            # Track strategies at specified intervals
            if track_strategies and i % self.track_frequency == 0:
                self._track_current_strategies(i)
        
        # Create node_dict for compatibility with existing notebooks
        self._create_node_dict()
    
    def _create_node_dict(self) -> None:
        """Create node_dict for compatibility with existing code"""
        self.node_dict = {}
        for key, info_set in self.info_sets.items():
            # Create a simple wrapper that has a policy() method
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
            
            # Handle case where strategy length doesn't match actions
            if len(strategy) != len(actions):
                return np.random.choice(actions)
            
            # Sample action according to strategy
            return np.random.choice(actions, p=strategy)
        else:
            # Random action if no strategy learned yet
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
            # Return uniform random if info set not found
            actions = self.game.available_actions()
            return np.ones(len(actions)) / len(actions)
        
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