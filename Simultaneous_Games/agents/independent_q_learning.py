from itertools import product
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class IQL(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID):
        super().__init__(game, agent)
        self.num_actions = self.game.num_actions(self.agent)
        self.q_table = dict()  # Dictionary to handle arbitrary state keys
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99
        self.last_state = None
        self.last_action = None

    def action(self):
        state = self._get_obs_key(self.game.observe(self.agent))
        self.last_state = state

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        self.update()

        if np.random.rand() < self.epsilon:
            self.last_action = np.random.choice(self.num_actions)
        else:
            self.last_action = np.argmax(self.q_table[state])

        return self.last_action

    def update(self):
        obs = self.get_obs()
        reward = self.get_reward()
        current_state = self._get_obs_key(obs)
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(self.num_actions)

        max_future_q = np.max(self.q_table[current_state])
        current_q = self.q_table[self.last_state][self.last_action]
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        self.q_table[self.last_state][self.last_action] += self.alpha * td_error
        
    def get_obs(self):
        """Returns the current observation of the agent."""
        return self.game.observe(self.agent)
    
    def get_reward(self):
        """Returns the current reward of the agent."""
        if self.game.rewards[self.agent] is None:
            return 0
        return self.game.rewards[self.agent]

    def _get_obs_key(self, obs):
        """Turns the observation dictionary into a tuple usable as a Q-table key."""
        # You might want to define a more compact state representation depending on your observation space
        if obs is None:
            return tuple(np.zeros(self.num_actions, dtype=int)) # Esto esta mal en realidad pero bue
        return tuple(obs)
