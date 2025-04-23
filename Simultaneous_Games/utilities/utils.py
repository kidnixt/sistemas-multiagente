import matplotlib.pyplot as plt
import numpy as np

def run_and_plot(agent_combination: dict, game, num_iterations=1000, title_suffix=""):
    """
    Ejecuta un experimento multi-agente y grafica la evolución de las políticas.
    Muestra todas las políticas en subplots horizontales del mismo tamaño.

    Parameters:
    - agent_combination: dict agent_id -> instancia del agente (FictitiousPlay, RegretMatching, etc.)
    - game: instancia del juego (con método .step() y .agents)
    - num_iterations: cantidad de pasos a simular
    - title_suffix: string opcional para agregar al título del gráfico (ej: "FP vs RM")

    Returns:
    - policies: dict agent_id -> np.array (iteraciones x acciones)
    """
    game.reset()
    agents = list(agent_combination.keys())
    num_agents = len(agents)

    # Inicializar historial de políticas
    policies = {agent: [agent_combination[agent].policy().copy()] for agent in agents}
    action_history = {agent: [] for agent in agents}


    for _ in range(num_iterations):
        actions = {agent: agent_combination[agent].action() for agent in agents}
        game.step(actions)
        for agent in agents:
            policies[agent].append(agent_combination[agent].policy().copy())
            action_history[agent].append(actions[agent])


    # Convertir listas a arrays
    for agent in policies:
        policies[agent] = np.array(policies[agent])

    # Crear figura con subplots horizontales
    fig, axs = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5), sharey=True)

    if num_agents == 1:
        axs = [axs]  # manejar caso especial con un solo subplot

    for idx, agent in enumerate(agents):
        policy_matrix = policies[agent]
        ax = axs[idx]
        for action in range(policy_matrix.shape[1]):
            ax.plot(
                range(num_iterations + 1),
                policy_matrix[:, action],
                label=f'Action {action + 1}',
                marker='o',
                linewidth=1
            )
        ax.set_title(f'{agent} {title_suffix}')
        ax.set_xlabel('Iteration')
        if idx == 0:
            ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return policies, action_history



import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_to_nash(policies: dict, nash_equilibrium: dict, title_suffix=""):
    """
    Grafica la diferencia por acción respecto al equilibrio de Nash, usando subplots horizontales.

    Parameters:
    - policies: dict[agent_id] -> np.array (iteraciones x acciones)
    - nash_equilibrium: dict[agent_id] -> lista con probabilidad de cada acción (ej: [0.5, 0.5])
    - title_suffix: string adicional para el título
    """

    agents = list(policies.keys())
    num_agents = len(agents)

    fig, axs = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5), sharey=True)

    if num_agents == 1:
        axs = [axs]

    for idx, agent in enumerate(agents):
        policy_matrix = policies[agent]
        nash = np.array(nash_equilibrium[agent])
        diffs = np.abs(policy_matrix - nash)  # shape: (T, A)

        ax = axs[idx]
        for action in range(diffs.shape[1]):
            ax.plot(
                range(len(diffs)),
                diffs[:, action],
                label=f'Action {action}',
                linewidth=2
            )
        ax.set_title(f'Convergence for {agent} {title_suffix}')
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Abs. Error per Action")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_action_trace_from_history(action_history: dict, title_suffix=""):
    """
    Grafica la secuencia de acciones reales que fueron jugadas por cada agente en subplots horizontales.

    Parameters:
    - action_history: dict[agent_id] -> lista de acciones jugadas en cada iteración
    - title_suffix: texto adicional para el título
    """
    agents = list(action_history.keys())
    num_agents = len(agents)

    fig, axs = plt.subplots(1, num_agents, figsize=(6 * num_agents, 3), sharey=True)

    if num_agents == 1:
        axs = [axs]

    for idx, agent in enumerate(agents):
        actions = action_history[agent]
        ax = axs[idx]
        ax.plot(actions, label=f"{agent}", marker='o')
        ax.set_title(f"{agent} {title_suffix}")
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Action")
        ax.set_yticks(sorted(set(actions)))
        ax.grid(True)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_action_histogram(policies: dict, title_suffix=""):
    """
    Muestra un histograma con la frecuencia promedio de acciones de cada agente, en subplots horizontales.

    Parameters:
    - policies: dict[agent_id] -> np.array (iteraciones x acciones)
    - title_suffix: string adicional para el título
    """

    agents = list(policies.keys())
    num_agents = len(agents)

    fig, axs = plt.subplots(1, num_agents, figsize=(5 * num_agents, 4), sharey=True)

    if num_agents == 1:
        axs = [axs]

    for idx, agent in enumerate(agents):
        avg_policy = np.mean(policies[agent], axis=0)
        ax = axs[idx]
        ax.bar(range(len(avg_policy)), avg_policy)
        ax.set_title(f"{agent} {title_suffix}")
        ax.set_xlabel("Action")
        if idx == 0:
            ax.set_ylabel("Avg. Probability")
        ax.set_xticks(range(len(avg_policy)))
        ax.grid(axis='y')

    plt.tight_layout()
    plt.show()




def plot_dual_agent_simplex(empirical_distributions: dict, title="Empirical distributions in dual simplex"):
    import matplotlib.pyplot as plt
    import numpy as np

    assert len(empirical_distributions) == 2, "Debe haber exactamente dos agentes."

    agent_names = list(empirical_distributions.keys())
    colors = ['steelblue', 'darkorange']
    n_points = len(next(iter(empirical_distributions.values())))

    plt.figure(figsize=(7, 7))

    # Diagonal compartida (π₃ = 0)
    plt.plot([0, 1], [1, 0], 'k--', linewidth=1.2, label="π₃ = 0")

    for i, agent in enumerate(agent_names):
        traj = np.array(empirical_distributions[agent])
        rock = traj[:, 0]
        paper = traj[:, 1]
        scissors = 1 - rock - paper

        valid = (rock >= 0) & (paper >= 0) & (scissors >= 0)
        rock, paper = rock[valid], paper[valid]

        # Reflejar agente 2 sobre la diagonal
        if i == 1:
            rock, paper = 1 - paper, 1 - rock

        # Gradiente de alpha
        alphas = np.linspace(0.2, 1.0, len(rock))
        for j in range(1, len(rock)):
            plt.plot(
                [rock[j - 1], rock[j]], [paper[j - 1], paper[j]],
                color=colors[i], alpha=alphas[j], lw=2
            )

        # Puntos discretos cada 10 pasos
        plt.scatter(rock[::10], paper[::10], color=colors[i], s=15, alpha=0.6, label=f"{agent}")

        # Equilibrio Nash
        x_eq, y_eq = (1/3, 1/3)
        if i == 1:
            x_eq, y_eq = 1 - y_eq, 1 - x_eq
        plt.plot(x_eq, y_eq, 'k*', markersize=14)

    # Ejes y estilo
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')

    # Ejes primarios (agente 1)
    plt.xlabel(r"$\pi_1(\mathrm{Rock})$", loc='center')
    plt.ylabel(r"$\pi_1(\mathrm{Paper})$", loc='center')

    # Ejes secundarios (agente 2)
    ax2 = plt.gca().secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))
    ax2.set_xticks(ticks)
    ax2.set_xlabel(r"$\pi_2(\mathrm{Rock})$", loc='center')

    ax3 = plt.gca().secondary_yaxis('right', functions=(lambda y: 1 - y, lambda y: 1 - y))
    ax3.set_yticks(ticks)
    ax3.set_ylabel(r"$\pi_2(\mathrm{Paper})$", loc='center')

    # Leyenda y título
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()




def compute_empirical_distributions(action_history: dict, num_actions=3):
    emp = {}
    for agent, actions in action_history.items():
        counts = np.zeros(num_actions)
        traj = []
        for a in actions:
            counts[a] += 1
            freq = counts / np.sum(counts)
            traj.append(freq.copy())
        emp[agent] = traj
    return emp

