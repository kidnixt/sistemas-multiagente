import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from agents.jal_am import JALAM
from agents.independent_q_learning import IQL
from games.foraging import Foraging

def run(game, agents, verbose=False, render=False, training=True):
    game.reset()
    cum_rewards = dict(map(lambda agent: (agent, 0), game.agents))

    if render:
        game.render()
        time.sleep(2) # Wait for 2 seconds before starting the game

    while not game.done():
        actions = dict(map(lambda agent: (agent, agents[agent].action()), game.agents))
        game.step(actions)

        for agent in game.agents:
            if training:
                # Update the agent only if training is enabled
                agents[agent].update(actions)
            cum_rewards[agent] += game.reward(agent)

        if verbose:
            for agent in game.agents:
                    print(f"Agent {agent} reward: {game.reward(agent)}")
                    print(f"Agent {agent} observe: {game.observe(agent)}")
            
        if render:
            game.render()
            time.sleep(0.1)
        
    return cum_rewards

def train(game, agents, iterations, episodes_per_iteration, verbose=False):
    reward_list = {agent: [] for agent in game.agents}
    rewards_per_iteration = {agent: [] for agent in game.agents}
    for i in range(iterations):
        for j in range(episodes_per_iteration):
            cum_rewards = run(game, agents, verbose=False, render=False, training=True)
            for agent in game.agents:
                reward_list[agent].append(cum_rewards[agent])

        for agent in game.agents:
            rewards_per_iteration[agent].append(np.mean(reward_list[agent][-episodes_per_iteration:]))
        if verbose:
            # Print the average rewards for each agent after each iteration
            print(f"Iteration {i+1}, Total Episodes {(j+1)* (i+1)}")
            for agent in game.agents:
                print(f"Agent {agent}, Average reward: {rewards_per_iteration[agent][i]}")
    
    # return the average rewards for each agent after training
    return rewards_per_iteration

def plot_rewards(rewards, config):
    plt.figure(figsize=(10, 5))
    for agent in rewards:
        plt.plot(rewards[agent], label=f"Agent {agent}")
    plt.title(f"Rewards for Configuration {config['game']}")
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rewards_on_multiple_runs(all_rewards, config):
    # Convert to 3D array: runs x agents x iterations
    all_rewards = np.array(all_rewards)

    # Compute mean and std across runs, for each agent and iteration
    mean_rewards = np.mean(all_rewards, axis=0)  # shape: agents x iterations
    std_rewards = np.std(all_rewards, axis=0)    # shape: agents x iterations

    # Plot mean and std for each agent
    iterations = np.arange(mean_rewards.shape[1])
    for agent_idx, agent in enumerate(np.arange(mean_rewards.shape[0])):
        plt.plot(iterations, mean_rewards[agent_idx], label=f"Agent {agent} Mean")
        plt.fill_between(iterations, 
                        mean_rewards[agent_idx] - std_rewards[agent_idx], 
                        mean_rewards[agent_idx] + std_rewards[agent_idx], 
                        alpha=0.2, label=f"Agent {agent} Std")

    plt.title(f"Mean and Std Rewards over {config['num_runs']} Runs for {config['game']}")
    plt.xlabel("Iterations")
    plt.ylabel("Rewards")
    plt.legend()
    plt.grid()
    plt.show()

def plot_sum_all_rewards(all_rewards, config):
    # all_rewards shape: (num_runs, num_agents, num_iterations)
    # Compute sum of rewards across agents for each run and iteration
    sum_rewards_per_run = np.sum(all_rewards, axis=1)  # shape: (num_runs, num_iterations)

    # Compute mean and std across runs, for each iteration
    mean_sum_rewards = np.mean(sum_rewards_per_run, axis=0)  # shape: (num_iterations,)
    std_sum_rewards = np.std(sum_rewards_per_run, axis=0)    # shape: (num_iterations,)

    # Plot mean and std of sum of rewards
    iterations = np.arange(mean_sum_rewards.shape[0])
    plt.plot(iterations, mean_sum_rewards, label="Mean Sum of Rewards")
    plt.fill_between(iterations, 
                    mean_sum_rewards - std_sum_rewards, 
                    mean_sum_rewards + std_sum_rewards, 
                    alpha=0.2, label="Std Dev")
    plt.axhline(y=1, color='r', linestyle='--', label="Target")
    plt.title(f"Mean and Std of Sum of Rewards over {config['num_runs']} Runs for {config['game']}")
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Rewards")
    plt.legend()
    plt.grid()
    plt.show()

def epsilon_func(i: int, min_value=0.05, max_value=1.0, decay_steps=8000):
        """
        Epsilon decay should linearly decay from max_value to min_value in the first decay_steps,
        and then stay at min_value for the rest of the training.
        """
        if i < decay_steps:
            # Linearly decay epsilon
            return max_value - (max_value - min_value) * (i / decay_steps)
        else:
            # Keep epsilon at min_value after decay_steps
            return min_value
        
def get_epsilon_func(min_value=0.05, max_value=1.0, decay_steps=8000):
    """
    Returns the epsilon function with the specified parameters.
    """
    return partial(epsilon_func, min_value=min_value, max_value=max_value, decay_steps=decay_steps)

# Alpha is just 0.1 for all steps
def alpha_func(i: int, value=0.1):
        return value

def get_alpha_func(value =0.1):
    """
    Returns the alpha function with the specified parameters.
    """
    return partial(alpha_func, value=value)

def single_run(game, i, config):
    agents = {agent: JALAM(game, agent, alpha_func=config["agent"]["alpha"], epsilon_func=config["agent"]["epsilon"], 
                             gamma=config["agent"]["gamma"], seed=i)
                    for agent in game.agents
                }
    rewards = train(game, agents, config["train"]["iterations"], config["train"]["episodes_per_iteration"], verbose=False)
    rewards_matrix = np.array([rewards[agent] for agent in game.agents])
    return rewards_matrix

def single_run_iql(game, i, config):
    agents = {agent: IQL(game, agent, alpha_func=config["agent"]["alpha"], epsilon_func=config["agent"]["epsilon"], 
                             gamma=config["agent"]["gamma"], seed=i)
                    for agent in game.agents
                }
    print(len(agents))
    rewards = train(game, agents, config["train"]["iterations"], config["train"]["episodes_per_iteration"], verbose=False)
    rewards_matrix = np.array([rewards[agent] for agent in game.agents])
    return rewards_matrix