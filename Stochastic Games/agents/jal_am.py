import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID
from collections import defaultdict

class JALAM(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, gamma=0.9, alpha_func=None, epsilon_func=None, seed=None):
        """
        Joint Action Learning Agent Modeling (JALAM) agent.

        Args:
            game (SimultaneousGame): The game instance.
            agent (AgentID): The ID of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
        """
        super().__init__(game, agent)
        self.gamma = gamma
        
        self.iteration = 0
        if alpha_func is not None:
            self._alpha_value = alpha_func
        if epsilon_func is not None:
            self._epsilon_value = epsilon_func

        # Initialize Q-table for joint actions
        self.q_table = {}

        # Initialize a model of other agents' behavior
        self.other_agents = [a for a in game.agents if a != agent]
        self.action_probabilities = {
            other_agent: defaultdict(lambda: np.ones(game.num_actions(other_agent)) / game.num_actions(other_agent))
            for other_agent in self.other_agents
        }

        self.last_state = None
        self.last_action = None

        self.rng = np.random.default_rng(seed) 

    def action(self):
        """
        Select an action using an epsilon-greedy policy.
        """
        state = self._get_state_key(self.game.observe(self.agent))
        self.last_state = state

        # Initialize Q-values for the state if not already present
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.game.num_actions(self.agent))

        # Epsilon-greedy action selection
        if self.rng.random() < self._epsilon_value(self.iteration):
            self.last_action = self.rng.choice(self.game.num_actions(self.agent))
        else:
            self.last_action = self.rng.choice(np.where(self.q_table[state] == np.max(self.q_table[state]))[0])

        return self.last_action

    def update(self, actions):
        self.iteration += 1
        reward = self.game.reward(self.agent)
        next_state = self._get_state_key(self.game.observe(self.agent))

        # Update the model of other agents' behavior
        self._update_other_agents(actions, next_state)

        # Initialize Q-values for the next state if not already present
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.game.num_actions(self.agent))

        # --- Compute AV for each possible action in next_state ---
        other_indices = np.array([self.game.agents.index(a) for a in self.other_agents])
        num_actions = self.game.num_actions(self.agent)
        av_values = np.zeros(num_actions)

        # Precompute all action probabilities for other agents at next_state
        other_probs = []
        for agent in self.other_agents:
            probs = self.action_probabilities[agent].get(next_state)
            if probs is None:
                probs = np.ones(self.game.num_actions(agent)) / self.game.num_actions(agent)
            other_probs.append(probs)
        other_probs = np.array(other_probs, dtype=object)  # shape: (num_other_agents, num_actions_per_agent)

        for joint_action in self._get_joint_actions():
            a_i = joint_action[self.game.agents.index(self.agent)]
            # Get the actions of other agents for this joint action
            other_actions = np.array(joint_action)[other_indices]
            # Use numpy to get the probability for each other agent's action
            probs = np.array([other_probs[i][act] for i, act in enumerate(other_actions)])
            prob = probs.prod()
            av_values[a_i] += prob * self.q_table[next_state][a_i]
        max_av = np.max(av_values)

        current_q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] += self._alpha_value(self.iteration) * (
            reward + self.gamma * max_av - current_q
        )

    def _update_other_agents(self, actions, state):
        """
        Update the model of other agents' behavior based on their observed actions.

        Args:
            actions (dict): A dictionary mapping each agent to their last action.
        """
        for other_agent in self.other_agents:
            action = actions.get(other_agent)  # Get the action of the other agent from the dictionary
            if action is not None:
                num_actions = self.game.num_actions(other_agent)
                probabilities = self.action_probabilities[other_agent][state]
                self.action_probabilities[other_agent][state] = probabilities / probabilities.sum()

    def _get_state_key(self, obs):
        """
        Convert the observation into a hashable state key.
        """
        return tuple(obs)
    
    def _get_joint_actions(self):
        """
        Generate all possible joint actions for the agents.
        """
        from itertools import product
        actions = [range(self.game.num_actions(agent)) for agent in self.game.agents]
        return list(product(*actions))
    
    def _alpha_value(self, i: int):
        return 0.1 # Default value for alpha

    def _epsilon_value(self, i: int):
        return 0.5 # Default value for epsilon