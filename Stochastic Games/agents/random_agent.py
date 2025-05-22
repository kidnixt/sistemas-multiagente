import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class StochasticRandomAgent(Agent):
    def __init__(self, game: SimultaneousGame, agent: AgentID, seed=None):
        """Random agent for stochastic games.

        Args:
            game (SimultaneousGame): The game instance.
            agent (AgentID): The ID of the agent.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(game, agent)
        self.num_actions = self.game.num_actions(self.agent)
        self.rng = np.random.default_rng(seed)  # Random number generator for reproducibility

    def action(self):
        """Selects a random action from the available actions."""
        return self.rng.choice(self.num_actions)
    
    def update(self, actions):
        """Update function for the random agent. This is a no-op for random agents."""
        pass