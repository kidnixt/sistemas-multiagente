from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional
import copy

class InfoSet:
    def __init__(self, num_actions: int):
        self.regrets = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.current_strategy = np.ones(num_actions) / num_actions
        
    def get_strategy(self) -> np.ndarray:
        """Get current strategy using regret matching"""
        regret_sum = np.sum(np.maximum(self.regrets, 0))
        if regret_sum > 0:
            self.current_strategy = np.maximum(self.regrets, 0) / regret_sum
        else:
            self.current_strategy = np.ones(len(self.regrets)) / len(self.regrets)
        return self.current_strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        norm_sum = np.sum(self.strategy_sum)
        if norm_sum > 0:
            return self.strategy_sum / norm_sum
        else:
            return np.ones(len(self.strategy_sum)) / len(self.strategy_sum)

class CounterFactualRegret(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, seed: Optional[int] = None):
        super().__init__(game, agent)
        self.info_sets: Dict[str, InfoSet] = {}
        self.node_dict = {}  # For compatibility with notebook expectations
        
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
        """
        Counterfactual Regret Minimization algorithm
        
        Args:
            game_state: Current game state
            player: Current player
            reach_prob: Reach probabilities for each player
            chance_reach: Reach probability from chance events
            update_player: Player whose regrets we're updating (None means update current player)
        """
        
        # Terminal node
        if game_state.terminated():
            return game_state.reward(self.agent)
        
        # Get current player
        current_player = game_state.agent_selection
        if update_player is None:
            update_player = current_player
            
        # Get available actions
        actions = game_state.available_actions()
        num_actions = len(actions)
        
        # Get information set
        info_set_key = self.get_info_set_key(game_state)
        info_set = self.get_or_create_info_set(info_set_key, num_actions)
        
        # Get current strategy
        strategy = info_set.get_strategy()
        
        # Compute utilities for each action
        action_utilities = np.zeros(num_actions)
        node_utility = 0.0
        
        for i, action in enumerate(actions):
            # Create new game state
            new_game_state = game_state.clone()
            new_game_state.step(action)
            
            # Update reach probabilities
            new_reach_prob = reach_prob.copy()
            if current_player in new_reach_prob:
                new_reach_prob[current_player] *= strategy[i]
            
            # Recursive call
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
            # Calculate counterfactual reach (reach probability of other players)
            cfr_reach = 1.0
            for p, prob in reach_prob.items():
                if p != current_player:
                    cfr_reach *= prob
            cfr_reach *= chance_reach
            
            # Update regrets
            for i in range(num_actions):
                regret = action_utilities[i] - node_utility
                info_set.regrets[i] += cfr_reach * regret
            
            # Update strategy sum
            my_reach = reach_prob.get(current_player, 1.0)
            for i in range(num_actions):
                info_set.strategy_sum[i] += my_reach * strategy[i]
        
        return node_utility
    
    def train(self, iterations: int = 10000) -> None:
        """Train the agent using CFR"""            
        for i in range(iterations):
            # Train for each player
            for player in self.game.agents:
                self.game.reset()
                reach_prob = {p: 1.0 for p in self.game.agents}
                self.cfr(self.game, self.game.agent_selection, reach_prob, 1.0, player)
        
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