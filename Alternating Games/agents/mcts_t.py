from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable, Optional
import sys
import random


class MCTSNode:
    """A node in the Monte‑Carlo search tree."""

    def __init__(self, parent: 'MCTSNode | None', game: AlternatingGame, action: Optional[ActionType]):
        self.parent: 'MCTSNode | None' = parent
        self.game: AlternatingGame = game          # cloned game state at this node
        self.action: Optional[ActionType] = action  # action that led to this node (None for root)
        self.children: list[MCTSNode] = []          # child nodes
        self.visits: int = 0                       # visit count N(s)
        self.cum_rewards: np.ndarray = np.zeros(len(game.agents), dtype=float)  # cumulative returns

    # ------------------------------------------------------------------ #
    def is_fully_expanded(self) -> bool:
        """All legal actions already have child nodes."""
        return len(self.children) == len(self.game.available_actions())


# ---------------------------------------------------------------------- #
#  Upper‑Confidence Bound (UCB1) used for child selection                #
# ---------------------------------------------------------------------- #

def ucb(node: MCTSNode, max_agent: AgentID, C: float = sqrt(2)) -> float:
    """Upper‑Confidence Bound for Trees (UCT) score, seen from max_agent.

    Works for *two‑player, zero‑sum* games.  The exploitation term is the
    average reward of `max_agent` at that node; the exploration term biases
    toward less‑visited nodes.
    """
    if node.visits == 0:
        return float("inf")

    # Exploitation (average reward for the searching player)
    idx = node.game.agent_name_mapping[max_agent]
    avg_reward = node.cum_rewards[idx] / node.visits

    # Exploration (sqrt(log N_parent / N_node))
    parent_visits = node.parent.visits if node.parent else 1  # root: log(1)=0
    exploration = C * sqrt(log(parent_visits) / node.visits)

    return avg_reward + exploration


def uct(node: MCTSNode, max_agent: AgentID) -> MCTSNode:
    """Select the child with highest UCB score."""
    return max(node.children, key=lambda child: ucb(child, max_agent))


# ---------------------------------------------------------------------- #
#  Main agent                                                            #
# ---------------------------------------------------------------------- #

class MonteCarloTreeSearch(Agent):
    """Generic MCTS agent for two‑player, zero‑sum, alternating games."""

    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        simulations: int = 200,
        rollouts: int = 8,
        depth: Optional[int] = None,
        selection_policy: Callable[[MCTSNode, AgentID], MCTSNode] = uct,
    ) -> None:
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.depth = depth
        self.selection_policy = selection_policy

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def action(self) -> ActionType:
        best_action, _ = self._mcts()
        return best_action

    # ------------------------------------------------------------------ #
    # Core loop                                                          #
    # ------------------------------------------------------------------ #
    def _mcts(self) -> tuple[ActionType, float]:
        root = MCTSNode(parent=None, game=self.game.clone(), action=None)

        for _ in range(self.simulations):
            # 1. Selection
            leaf = self._select(root)
            # 2. Expansion
            expanded = self._expand(leaf)
            # 3. Simulation / roll‑out
            rewards = self._rollout(expanded)
            # 4. Back‑propagation
            self._backpropagate(expanded, rewards)

        return self._best_child_action(root)

    # ------------------------------------------------------------------ #
    # 1. Selection                                                        #
    # ------------------------------------------------------------------ #
    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.game.terminated() and current.is_fully_expanded():
            current = self.selection_policy(current, self.agent)
        return current

    # ------------------------------------------------------------------ #
    # 2. Expansion                                                        #
    # ------------------------------------------------------------------ #
    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.game.terminated():
            return node

        explored_actions = {child.action for child in node.children}
        for action in node.game.available_actions():
            if action not in explored_actions:
                next_game = node.game.clone()
                next_game.step(action)
                child = MCTSNode(parent=node, game=next_game, action=action)
                node.children.append(child)
                return child

        # Should not happen (node fully expanded)
        return node

    # ------------------------------------------------------------------ #
    # 3. Simulation / roll‑out                                            #
    # ------------------------------------------------------------------ #
    def _rollout(self, node: MCTSNode) -> np.ndarray:
        cumulative = np.zeros(len(self.game.agents), dtype=float)

        for _ in range(self.rollouts):
            sim_game = node.game.clone()
            remaining = self.depth if self.depth is not None else sys.maxsize

            while not sim_game.terminated() and remaining > 0:
                remaining -= 1
                actions = sim_game.available_actions()
                sim_game.step(random.choice(actions))

            # Terminal or depth‑limit reward
            if sim_game.terminated():
                for i, aid in enumerate(self.game.agents):
                    cumulative[i] += sim_game.reward(aid)
            else:  # depth limit reached but not terminal
                for i, aid in enumerate(self.game.agents):
                    cumulative[i] += sim_game.eval(aid)

        return cumulative / self.rollouts

    # ------------------------------------------------------------------ #
    # 4. Back‑propagation                                                 #
    # ------------------------------------------------------------------ #
    def _backpropagate(self, node: MCTSNode, rewards: np.ndarray) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.cum_rewards += rewards
            current = current.parent

    # ------------------------------------------------------------------ #
    # Best action for the root                                            #
    # ------------------------------------------------------------------ #
    def _best_child_action(self, root: MCTSNode) -> tuple[ActionType, float]:
        if not root.children:
            return None, 0.0

        idx = root.game.agent_name_mapping[self.agent]
        best = max(root.children, key=lambda c: c.cum_rewards[idx] / c.visits)
        value = best.cum_rewards[idx] / best.visits
        return best.action, value
