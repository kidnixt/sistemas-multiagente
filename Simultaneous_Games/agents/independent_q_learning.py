from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class IQL(Agent):
    def __init__(self, game, agent):
        super().__init__(game, agent)
        self.q_table = np.zeros((8,8,3,8,8,3,8,8,3,game.num_actions(agent)))
        self.epsilon = 0.1
    
    def action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.game.num_actions(self.agent))
        else:
            return np.argmax(self.q_table.get(self.agent, np.zeros(self.game.num_actions(self.agent))))

    def update(self, obs, reward):
        # Update Q-table using the observed reward and the action taken
        state = tuple(obs.values())
        action = self.action()
        next_state = tuple(obs.values())
        self.q_table.get(self.agent, np.zeros(self.game.num_actions(self.agent)))[state][action] += 0.1 * td_error
        
