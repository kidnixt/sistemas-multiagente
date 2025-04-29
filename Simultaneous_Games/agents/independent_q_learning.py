from itertools import product
import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID

class IQL(Agent):
    def _init_(self, game: SimultaneousGame, agent: AgentID):
        super()._init_(game, agent)
        self.num_actions = self.game.num_actions(self.agent)
        self.q_table = dict()
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.alpha = 0.1
        self.gamma = 0.99
        self.last_state = None
        self.last_action = None

    def action(self):
        obs = self.get_obs()
        state = self._get_obs_key(obs)
        self.last_state = state

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        # Heurística: si estoy adyacente a una fruta → hago LOAD
        if self.is_adjacent_to_fruit(obs):
            self.last_action = 5  # 5 = LOAD
            return self.last_action

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
        return self.game.observe(self.agent)

    def get_reward(self):
        """Reward shaping mejorado: según posibilidad real de cargar."""
        rewards = list(self.game.rewards.values())
        r_total = sum(r if r is not None else 0 for r in rewards)

        if r_total > 0:
            return r_total

        obs = self.get_obs()
        if obs is None:
            return 0

        fx, fy = int(obs[0]), int(obs[1])
        food_lvl = int(obs[2])
        x_self, y_self = int(obs[3]), int(obs[4])
        self_lvl = int(obs[5])
        x_other, y_other = int(obs[6]), int(obs[7])

        # Verificar adyacencia
        def is_adjacent(ax, ay, bx, by):
            return (abs(ax - bx) == 1 and ay == by) or (abs(ay - by) == 1 and ax == bx)

        self_adj = is_adjacent(x_self, y_self, fx, fy)
        other_adj = is_adjacent(x_other, y_other, fx, fy)

        if self_adj and other_adj:
            return 0.1  # ambos cerca → recompensa grande
        elif self_adj:
            if self_lvl >= food_lvl:
                return 0.1  # puedo cargar solo → recompensa grande
            else:
                return 0.05  # estoy bien posicionado pero necesito ayuda
        else:
            return -0.01  # castigo leve por no contribuir

    def is_adjacent_to_fruit(self, obs):
        """Chequea si estoy adyacente a una fruta (sin diagonal)."""
        if obs is None:
            return False
        fx, fy = int(obs[0]), int(obs[1])
        x_self, y_self = int(obs[3]), int(obs[4])
        return (abs(fx - x_self) == 1 and fy == y_self) or (abs(fy - y_self) == 1 and fx == x_self)

    @staticmethod
    def simplify_obs(obs):
        if obs is None:
            return (0,) * 6

        obs = list(map(int, obs))
        
        # Asumimos que cada fruta tiene 3 valores: x, y, nivel
        num_fruits = (len(obs) - 6) // 3

        self_x, self_y, self_lvl = obs[-6], obs[-5], obs[-4]
        other_x, other_y, other_lvl = obs[-3], obs[-2], obs[-1]

        def is_adjacent(ax, ay, bx, by):
            return (abs(ax - bx) == 1 and ay == by) or (abs(ay - by) == 1 and ax == bx)

        def discretize_level(lvl):
            return 0 if lvl == 1 else 1 if lvl == 2 else 2

        best_fruit = None
        min_distance = float('inf')

        for i in range(num_fruits):
            fx = obs[i * 3]
            fy = obs[i * 3 + 1]
            flvl = obs[i * 3 + 2]

            if self_lvl >= flvl:
                dist = abs(fx - self_x) + abs(fy - self_y)
                if dist < min_distance:
                    min_distance = dist
                    best_fruit = (fx, fy, flvl)

        # Si no hay fruta que pueda recolectar solo, tomamos la más cercana igual
        if best_fruit is None:
            for i in range(num_fruits):
                fx = obs[i * 3]
                fy = obs[i * 3 + 1]
                flvl = obs[i * 3 + 2]

                dist = abs(fx - self_x) + abs(fy - self_y)
                if dist < min_distance:
                    min_distance = dist
                    best_fruit = (fx, fy, flvl)

        fx, fy, food_lvl = best_fruit

        self_adj = int(is_adjacent(self_x, self_y, fx, fy))
        other_adj = int(is_adjacent(other_x, other_y, fx, fy))
        can_load_alone = int(self_lvl >= food_lvl)

        return (
            self_adj,
            other_adj,
            discretize_level(food_lvl),
            discretize_level(self_lvl),
            discretize_level(other_lvl),
            can_load_alone,
        )


    def _get_obs_key(self, obs):
        return self.simplify_obs(obs)