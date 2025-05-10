import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID
import math

class IQL(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, q_table = None, gamma = 0.99, alpha_func = None, epsilon_func = None, start_random_q_table = True):
        """Independent Q-Learning agent for simultaneous games.
        
        Args:

            game (SimultaneousGame): The game instance.
            agent (AgentID): The ID of the agent.
            q_table (dict): The Q-table for the agent. If None, a new one will be created.
            gamma (float): Discount factor for future rewards.
            alpha_func (function): Function to compute the learning rate.
            epsilon_func (function): Function to compute the exploration rate.
            start_random_q_table (bool): If True, initialize Q-values randomly. Else, initialize to zero.
        """


        super().__init__(game, agent)
        self.num_actions = self.game.num_actions(self.agent)
        self.q_table = q_table if q_table is not None else dict()  # Dictionary to handle arbitrary state keys
        self.gamma = gamma
        
        if alpha_func is not None:
            self._alpha_value = alpha_func
        if epsilon_func is not None:
            self._epsilon_value = epsilon_func
        
        self.last_state = None
        self.last_action = None
        self.iteration = 0
        self.q_table_random = start_random_q_table

    def action(self):
        state = self._get_obs_key(self.game.observe(self.agent))
        self.last_state = state

        if state not in self.q_table:
            if self.q_table_random:
                self.q_table[state] = np.random.rand(self.num_actions)
            else:
                # Initialize Q-values to zero
                self.q_table[state] = np.zeros(self.num_actions)

        self.iteration += 1

        if np.random.rand() < self._epsilon_value(self.iteration):
            self.last_action = np.random.choice(self.num_actions)
        else:
            self.last_action = np.argmax(self.q_table[state])

        return self.last_action

    def update(self):
        obs = self.get_obs()
        reward = self.get_reward()
        current_state = self._get_obs_key(obs)
        if current_state not in self.q_table:
            self.q_table[current_state] = np.random.rand(self.num_actions)  # Initialize with random values

        max_future_q = np.max(self.q_table[current_state])
        current_q = self.q_table[self.last_state][self.last_action]
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        self.q_table[self.last_state][self.last_action] += self._alpha_value(self.iteration) * td_error
        
    def get_obs(self):
        """Returns the current observation of the agent."""
        return self.game.observe(self.agent)
    
    def get_reward(self):
        """Returns the current reward of the agent."""
        return self.game.rewards[self.agent]

    def _get_obs_key(self, obs):
        """Turns the observation dictionary into a tuple usable as a Q-table key."""
        return tuple(obs)
    
    def _alpha_value(self, i: int):
        return 0.1 # Default value for alpha

    def _epsilon_value(self, i: int):
        return 0.5 # Default value for epsilon

        
