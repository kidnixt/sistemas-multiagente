import numpy as np
from base.game import SimultaneousGame

class CentralizedQLearning:
    def __init__(self, game: SimultaneousGame, alpha=0.1, gamma=0.99, epsilon=0.5):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Centralized Q-table
        self.last_state = None
        self.last_action = None

    def _get_centralized_state(self):
        """Combine observations of all agents into a centralized state."""
        observations = [np.atleast_1d(self.game.observe(agent)) for agent in self.game.agents]
        return tuple(np.concatenate(observations))

    def _get_joint_action(self):
        """Choose a joint action for all agents."""
        if np.random.rand() < self.epsilon:
            # Random joint action
            return tuple(np.random.choice(self.game.action_spaces[agent].n) for agent in self.game.agents)
        else:
            # Best joint action
            state = self.last_state
            if state not in self.q_table:
                self.q_table[state] = np.random.rand(np.prod([self.game.action_spaces[agent].n for agent in self.game.agents]))
            joint_action_index = np.argmax(self.q_table[state])
            return np.unravel_index(joint_action_index, [self.game.action_spaces[agent].n for agent in self.game.agents])

    def update(self):
        """Update the centralized Q-table."""
        current_state = self._get_centralized_state()
        reward = sum(self.game.rewards[agent] for agent in self.game.agents)  # Centralized reward

        if current_state not in self.q_table:
            self.q_table[current_state] = np.random.rand(np.prod([self.game.action_spaces[agent].n for agent in self.game.agents]))

        max_future_q = np.max(self.q_table[current_state])
        current_q = self.q_table[self.last_state][np.ravel_multi_index(self.last_action, [self.game.action_spaces[agent].n for agent in self.game.agents])]
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - current_q

        self.q_table[self.last_state][np.ravel_multi_index(self.last_action, [self.game.action_spaces[agent].n for agent in self.game.agents])] += self.alpha * td_error

    def step(self):
        """Perform a step in the environment."""
        self.last_state = self._get_centralized_state()
        self.last_action = self._get_joint_action()
        actions = {agent: action for agent, action in zip(self.game.agents, self.last_action)}
        self.game.step(actions)
        self.update()