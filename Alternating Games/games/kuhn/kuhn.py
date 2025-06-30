import numpy as np
from numpy import ndarray
from numpy import random
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from base.game import AlternatingGame, AgentID, ActionType

class KuhnPoker(AlternatingGame):

    def __init__(self, initial_player=None, seed=None, render_mode='human'):
        self.render_mode = render_mode

        self.seed = seed
        random.seed(seed)

        # agents
        self.agents = ["agent_0", "agent_1"]
        self.players = [0, 1]

        self.initial_player = initial_player

        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.agent_selection = None

        # actions
        self._moves = ['p', 'b']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # states
        self._max_moves = 3
        self._start = ''
        self._terminalset = set(['pp', 'pbp', 'pbb', 'bp', 'bb'])
        self._hist_space = Text(min_length=0, max_length=self._max_moves, charset=frozenset(self._moves))
        self._hist = None
        self._card_names = ['J', 'Q', 'K']
        self._num_cards = len(self._card_names)
        self._cards = list(range(self._num_cards))
        self._card_space = Discrete(self._num_cards)
        self._hand = None

        # observations
        self.observation_spaces = {
            agent: Dict({ 'card': self._card_space, 'hist': self._hist_space}) for agent in self.agents
        }
    
    def step(self, action: ActionType) -> None:
        agent = self.agent_selection
        # check for termination
        if (self.terminations[agent] or self.truncations[agent]):
            try:
                self._was_dead_step(action)
            except ValueError:
                print('Game has already finished - Call reset if you want to play again')
                return

        # perform step
        self._hist += self._moves[action]
        self._player = (self._player + 1) % 2
        self.agent_selection = self.agents[self._player]

        if self._hist in self._terminalset:
            history = self._hist
            first = self.initial_player
            second = 1 - first

            if history == 'pp':
                # showdown
                winner = np.argmax(self._hand)
                _rewards = [1 if p == winner else -1 for p in range(self.num_agents)]
            elif history == 'pbp':
                # player passes, second bets, first passes => second wins
                _rewards = [1 if p == second else -1 for p in range(self.num_agents)]
            elif history == 'bp':
                # first bets, second folds => first wins
                _rewards = [1 if p == first else -1 for p in range(self.num_agents)]
            elif history in ['pbb', 'bb']:
                # showdown with 2 chips
                winner = np.argmax(self._hand)
                _rewards = [2 if p == winner else -2 for p in range(self.num_agents)]
            else:
                raise ValueError(f"Unknown terminal history: {history}")
              
        
            self.rewards = dict(map(lambda p: (p, _rewards[self.agent_name_mapping[p]]), self.agents))
            self.terminations = dict(map(lambda p: (p, True), self.agents))

        self.observations = dict(map(lambda agent: (agent, {'card': self._hand[self.agent_name_mapping[agent]], 'hist': self._hist}), self.agents))

    def _set_initial(self):
        # set initial history
        self._hist = self._start

        # deal a card to each player
        random.seed(self.seed)
        self._hand = random.choice(self._cards, size=self.num_agents, replace=False)      

        # reset agent selection
        if self.initial_player is None:
            # select random player
            self.initial_player = random.choice(self.players)
 
        self._player = self.initial_player
        self.agent_selection = self.agents[self._player]

        
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._set_initial()

        self.observations = dict(map(lambda agent: (agent, {'card': self._hand[self.agent_name_mapping[agent]], 'hist': self._hist}), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._card_names[self._hand[self.agent_name_mapping[agent]]], self._hist)

    def observe(self, agent: AgentID) -> str:
        observation = str(self._hand[self.agent_name_mapping[agent]]) + self._hist
        return observation
    
    def available_actions(self):
        return list(range(self._num_actions))
    
    def random_change(self, agent: AgentID):
        agent_idx = self.agent_name_mapping[agent]
        agent_card = self._hand[agent_idx]
        other_idx = 1 - agent_idx 

        # Determine possible cards for the opponent
        possible_cards = [c for c in self._cards if c != agent_card]

        # Clone the game and assign a random one to the opponent
        new_game = self.clone()
        new_game._hand[agent_idx] = agent_card  # redundant, but explicit
        new_game._hand[other_idx] = np.random.choice(possible_cards)

        new_game.observations = dict(
        map(
            lambda agent: (
                agent,
                {'card': new_game._hand[new_game.agent_name_mapping[agent]], 'hist': new_game._hist}
            ),
            new_game.agents
        )
    )

        return new_game


    def action_move(self, action: ActionType) -> str:
        if action not in range(self._num_actions):
            raise ValueError(f"{action} is not a legal action.")
        
        return self._moves[action]