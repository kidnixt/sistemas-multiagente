from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from games.tictactoe import tictactoe_v3 as tictactoe
from base.game import AlternatingGame, AgentID
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class TicTacToe(AlternatingGame):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = tictactoe.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def reset(self):
        self.env.reset()
        self._update()

    def observe(self, agent: AgentID) -> ObsType:
        # A grid is list of lists, where each list represents a row
        # blank space = 0
        # agent = 1
        # opponent = 2
        # Ex:
        # [[0,0,2]
        #  [1,2,1]
        #  [2,1,0]]
        observation = self.env.observe(agent=agent)['observation']
        grid = np.sum(observation*[1,2], axis=2)
        return grid

    def step(self, action):
        self.env.step(action)
        self._update()

    def available_actions(self):
        return self.env.board.legal_moves()

    def render(self):
        print("Player:", self.agent_selection)
        print("Board:") 
        sq = np.array(self.env.board.squares).reshape((3, 3))
        for i in range(3):
            for j in range(3):
                if sq[i, j] == 0:
                    print(" . ", end="")
                elif sq[i, j] == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
            print()
        print()

    # def clone(self):
    #     self_clone = super().clone()
    #     self_clone.env.board.squares = self.env.board.squares.copy()
    #     self_clone.env.agent_selection = self.env.agent_selection
    #     return self_clone

    def clone(self):
        self_clone = TicTacToe(render_mode=self.env.render_mode)
        self_clone.env.board.squares = self.env.board.squares.copy()
        self_clone.env.rewards = self.env.rewards.copy()
        self_clone.env.terminations = self.env.terminations.copy()
        self_clone.env.truncations = self.env.truncations.copy()
        self_clone.env.infos = self_clone.env.infos.copy
        self_clone.env.agent_selection = self.env.agent_selection
        self_clone._update()
        return self_clone

    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")

        if self.terminated():
            return self.rewards[agent]

        grid = self.observe(agent)
        
        # Identificar jugadores (agent=2, opponent=1 según observe())
        my_piece = 2
        enemy_piece = 1
        
        my_score = 0
        enemy_score = 0
        
        # Evaluar todas las líneas (filas, columnas, diagonales)
        lines = []
        
        # Filas
        for i in range(3):
            lines.append(grid[i])
        
        # Columnas  
        for i in range(3):
            lines.append(grid[:, i])
        
        # Diagonales
        lines.append(grid.diagonal())
        lines.append(np.fliplr(grid).diagonal())
        
        # Evaluar cada línea
        for line in lines:
            my_count = np.sum(line == my_piece)
            enemy_count = np.sum(line == enemy_piece)
            empty_count = np.sum(line == 0)
            
            # Solo evaluar líneas no bloqueadas
            if my_count > 0 and enemy_count > 0:
                continue  # Línea bloqueada, sin valor
            
            if my_count == 2 and empty_count == 1:
                my_score += 50  # ¡Amenaza de victoria!
            elif my_count == 1 and empty_count == 2:
                my_score += 10  # Línea con potencial
            elif my_count == 0 and empty_count == 3:
                my_score += 1   # Línea disponible
                
            if enemy_count == 2 and empty_count == 1:
                enemy_score += 50  # ¡Amenaza enemiga!
            elif enemy_count == 1 and empty_count == 2:
                enemy_score += 10  # Línea enemiga con potencial
            elif enemy_count == 0 and empty_count == 3:
                enemy_score += 1   # Línea enemiga disponible
        
        # Bonus por centro (posición estratégica)
        if grid[1, 1] == my_piece:
            my_score += 5
        elif grid[1, 1] == enemy_piece:
            enemy_score += 5
        
        # Normalizar resultado
        total = my_score + enemy_score
        if total == 0:
            return 0.0
        
        return (my_score - enemy_score) / total

    
    
