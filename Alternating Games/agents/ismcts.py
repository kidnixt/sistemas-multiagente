from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable, Optional
import sys

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.visits = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = game.agent_selection


def ucb(node, C=sqrt(2)) -> float:
    if node.visits == 0:
        return float('inf')
    agent_idx = node.game.agent_name_mapping[node.agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits) / node.visits)


def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    return max(node.children, key=ucb)


class InformationSetMCTS(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int = 100, rollouts: int = 10,
                 depth: Optional[int] = None, selection: Callable[[MCTSNode, AgentID], MCTSNode] = uct) -> None:
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        self.depth = depth
        self.tree = {}

    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> tuple[ActionType, float]:
        for i in range(self.simulations):
            #print(f"Simulation {i+1}/{self.simulations}")
            determinized_game = self.determinize(self.game)
            root_key = self.get_info_key(determinized_game)
            if root_key not in self.tree:
                self.tree[root_key] = MCTSNode(parent=None, game=determinized_game, action=None)
            root = self.tree[root_key]

            node = self.select_node(root)
            node = self.expand_node(node)
            rewards = self.rollout(node)
            self.backprop(node, rewards)

        root_key = self.get_info_key(self.game)
        root = self.tree[root_key]
        return self.action_selection(root)

    def determinize(self, game: AlternatingGame) -> AlternatingGame:
        return game.random_change(self.agent)

    def get_info_key(self, game: AlternatingGame) -> tuple:
        return (game.observe(self.agent), game.agent_selection)

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while not curr_node.game.terminated():
            actions = curr_node.game.available_actions()
            if len(curr_node.children) < len(actions):
                return curr_node
            curr_node = self.selection(curr_node, self.agent)

        return curr_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        if node.game.terminated():
            return node
        actions = node.game.available_actions()
        if len(node.children) < len(actions):
            action = actions[len(node.children)]
            child_game = node.game.clone()
            child_game.step(action)
            info_key = self.get_info_key(child_game)
            if info_key in self.tree:
                child_node = self.tree[info_key]
            else:
                child_node = MCTSNode(parent=node, game=child_game, action=action)
                self.tree[info_key] = child_node
            node.children.append(child_node)
            return child_node
        return node

    def rollout(self, node: MCTSNode) -> np.ndarray:
        rewards = np.zeros(len(self.game.agents))
        for _ in range(self.rollouts):
            rollout_game = node.game.clone()
            depth = self.depth if self.depth is not None else sys.maxsize
            while not rollout_game.terminated() and depth > 0:
                depth -= 1
                actions = rollout_game.available_actions()
                action = np.random.choice(actions)
                rollout_game.step(action)
            for i, agent in enumerate(self.game.agents):
                rewards[i] += rollout_game.reward(agent) or 0
        return rewards / self.rollouts

    def backprop(self, node: MCTSNode, rewards: np.ndarray) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.cum_rewards += rewards
            current = current.parent

    def action_selection(self, node: MCTSNode) -> tuple[ActionType, float]:
        if not node.children:
            return None, 0
        agent_idx = self.game.agent_name_mapping[self.agent]
        best_child = max(
            node.children,
            key=lambda child: child.cum_rewards[agent_idx] / child.visits if child.visits > 0 else -float('inf')
        )
        return best_child.action, best_child.cum_rewards[agent_idx] / best_child.visits
