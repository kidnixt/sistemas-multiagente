from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    if node.visits == 0:
        return float('inf')  # Unvisited nodes get highest priority
    
    agent_idx = node.game.agent_name_mapping[node.agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> tuple[ActionType, float]:

        root = MCTSNode(parent=None, game=self.game.clone(), action=None)

        for i in range(self.simulations):
            node = root
            #node.game = self.game.clone()

            node = self.select_node(node=node)
            new_node = self.expand_node(node)

            rewards = self.rollout(new_node)
            self.backprop(new_node, rewards)

        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        current = node
        while current is not None:
            # Update visit count
            current.visits += 1
            
            # Update cumulative rewards for all agents
            current.cum_rewards += rewards
            
            # Move to parent node
            current = current.parent

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        
        # Perform multiple rollouts and average the results
        for _ in range(self.rollouts):
            # Clone the game state from the node
            rollout_game = node.game.clone()
            
            # Play randomly until the game terminates
            while not rollout_game.terminated():
                actions = rollout_game.available_actions()
                # Choose a random action
                random_action = np.random.choice(actions)
                rollout_game.step(random_action)
            
           # Get the final rewards and accumulate them
            for i, agent in enumerate(self.game.agents):
                rewards[i] += rollout_game.reward(agent)
        
        # Return average rewards across all rollouts
        return rewards / self.rollouts
    
    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while not curr_node.game.terminated():
            actions = curr_node.game.available_actions()
            
            # If node has unexplored actions, return it
            if len(curr_node.children) < len(actions):
                return curr_node
            
            # All actions have been expanded - select among existing children
            if curr_node.explored_children < len(curr_node.children):
                # Select next unvisited child
                curr_node = curr_node.children[curr_node.explored_children]
                curr_node.explored_children += 1
            else:
                # All children visited - use selection policy (UCT)
                curr_node = self.selection(curr_node, self.agent)
        
        return curr_node

    # !Esta version de select_node solo funciona si expand_node crea TODOS los hijos de cada nodo que expandimos
    # lo cual no es lo más eficiente. 
    # def select_node(self, node: MCTSNode) -> MCTSNode:
    #     curr_node = node
    #     while curr_node.children:
    #         if curr_node.explored_children < len(curr_node.children):
    #             # Select next unvisited child
    #             curr_node = curr_node.children[curr_node.explored_children]
    #             curr_node.explored_children += 1
    #         else:
    #             # All children visited - use selection policy (UCT)
    #             curr_node = self.selection(curr_node, self.agent)
    #     return curr_node

    def expand_node(self, node) -> MCTSNode:
        if node.game.terminated():
            return node
        
        actions = node.game.available_actions()
        
        # Only expand if there are unexplored actions
        if len(node.children) < len(actions):
            # Get the next action to expand
            action = actions[len(node.children)]
            
            # Create child for this action
            child_game = node.game.clone()
            child_game.step(action)
            child_node = MCTSNode(parent=node, game=child_game, action=action)
            node.children.append(child_node)
            return child_node  # Return the new child for immediate use
        
        return node

    def action_selection(self, node: MCTSNode) -> tuple[ActionType, float]:
        if not node.children:
            return None, 0
        
        # Get the agent index for this player
        agent_idx = self.game.agent_name_mapping[self.agent]
        
        # Find child with highest average reward for this agent
        best_child = max(node.children, 
                         key=lambda child: child.cum_rewards[agent_idx] / child.visits 
                         if child.visits > 0 else 0)
        
        action = best_child.action
        value = best_child.cum_rewards[agent_idx] / best_child.visits if best_child.visits > 0 else 0
        
        return action, value