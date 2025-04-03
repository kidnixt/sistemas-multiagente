import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict
from itertools import product


class RegretMatching(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        if (initial is None):
          self.curr_policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
          self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy() # Accion conjunta
        a = actions[self.agent]
        g = self.game.clone()
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        
        g.reset()
        g.step(actions)
        r_base = g.rewards[self.agent]

        # Calcular los rewards de agente para cada acción conjunta
        for act in range(g.num_actions(self.agent)):
            g.reset()
            actions[self.agent] = act
            g.step(actions)
            r = g.rewards[self.agent]
            u[act] = r - r_base
        
        return u
    
    def regret_matching(self):
        random = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        if np.sum(self.cum_regrets) <= 0:
            self.curr_policy = random
        else:
            self.curr_policy = np.maximum(self.cum_regrets,0)/np.sum(np.maximum(self.cum_regrets,0))

        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
           return
        regrets = self.regrets(actions)

        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))    

    def policy(self):
        return self.learned_policy