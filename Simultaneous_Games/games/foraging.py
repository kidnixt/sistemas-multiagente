from base.game import SimultaneousGame, ActionDict
import gymnasium as gym

class Foraging(SimultaneousGame):
    def __init__(self, config: None | str):
    
        # environment
        if config is None:
            config = "Foraging-8x8-2p-1f-v3"    
        self.env = gym.make(config)

        # agents
        self.agents = ["agent_" + str(r) for r in range(self.env.n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents()))))
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        # actions
        self.action_spaces = {
            agent: self.env.action_space[i] for i, agent in enumerate(self.agents)
        }   
            
    # num_agents
    def num_agents(self):
        return len(self.agents)
    
    # step
    def step(self, actions: ActionDict) -> tuple[dict, dict, dict, dict, dict]:
        # actions
        action = []
        for agent in self.agents:
            action.append(actions[agent])
        action = tuple(action)

        # step
        obs, rewards, done, truncated, info = self.env.step(action=action)

        # update observations, rewards, terminations, truncations, infos
        for i, agent in enumerate(self.agents):
            self.rewards[agent] = rewards[i]
            self.observations[agent] = obs[i]
            self.terminations[agent] = done
            self.truncations[agent] = truncated
            self.infos[agent] = info
        
        self._done = done
        self._truncated = truncated
        
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    # reset
    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))
        self._done = False
        self._truncated = False
    
    # render
    def render(self):
        self.env.render()   

    # close
    def close(self):
        self.env.close()

    # done
    def done(self):
        return self._done