from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class FictitiousPlay(Agent):
    
    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        
        if initial is not None:
            self.count = initial.copy()
        else:
            self.count: dict[AgentID, ndarray] = {}
            for agent in self.game.agents:
                self.count[agent] = np.random.randint(low=1, high=5, size=self.game.num_actions(agent))
                
        self.learned_policy: dict[AgentID, ndarray] = {}
        for agent in self.game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def get_rewards(self) -> dict:
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
    
        # Calcular los rewards de agente para cada acción conjunta
        for actions in product(*agents_actions):
            g.reset()
            g.step(dict(zip(g.agents, actions)))
            rewards[actions] = g.rewards[self.agent] 

        return rewards
    
    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        
        for action in range(self.game.num_actions(self.agent)):
            action_utility = 0
            for actions, reward in rewards.items(): # Todas las acciones conjuntas posibles
                relation = dict(zip(self.game.agents, actions)) # Relaciono para cada accion del conjunto el agente que la realiza
                if relation[self.agent] != action: # Si la acción de mi agente no es la acción actual, ignoro la accion conjunta
                    continue
                conjunction_prob = np.prod([self.learned_policy[agent][a] for agent, a in relation.items() if agent != self.agent]) # Calculo la probabilidad de la acción conjunta ignorando la acción del agente
                conjunction_utility = conjunction_prob * reward # Calculo la utilidad de la acción conjunta
                action_utility += conjunction_utility # Sumo la utilidad de la acción conjunta a la utilidad de la acción del agente
            
            utility[action] = action_utility

        return utility
    
    def bestresponse(self):
        a = np.argmax(self.get_utility())
        return a
     
    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()
    
    def policy(self):
       return self.learned_policy[self.agent]
    