import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Importar seaborn para obtener paletas de colores

def run_and_plot(agent_combination: dict, game, num_iterations=1000, title_suffix=""):
    """
    Ejecuta un experimento multi-agente y grafica la evolución de las políticas.
    Muestra todas las políticas en subplots horizontales del mismo tamaño,
    utilizando colores distintos para cada acción en cada subplot.

    Parameters:
    - agent_combination: dict agent_id -> instancia del agente (FictitiousPlay, RegretMatching, etc.)
    - game: instancia del juego (con método .step() y .agents)
    - num_iterations: cantidad de pasos a simular
    - title_suffix: string opcional para agregar al título del gráfico (ej: "FP vs RM")

    Returns:
    - policies: dict agent_id -> np.array (iteraciones x acciones)
    - action_history: dict agent_id -> list de acciones tomadas
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

    # Definir una paleta de colores lo suficientemente grande
    num_total_actions = max(policy.shape[1] for policy in policies.values())
    colors = sns.color_palette( "tab20",n_colors=num_total_actions)

    for idx, agent in enumerate(agents):
        policy_matrix = policies[agent]
        ax = axs[idx]
        num_actions = policy_matrix.shape[1]
        for action in range(num_actions):
            ax.plot(
                range(num_iterations + 1),
                policy_matrix[:, action],
                label=f'Action {action + 1}',
                marker='o',
                linewidth=1,
                color=colors[action % num_total_actions]  # Asignar color usando el índice de la acción
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
    Grafica la diferencia por acción respecto al equilibrio de Nash, usando la misma paleta de colores
    que run_and_plot ("tab20") para garantizar consistencia visual.

    Parameters:
    - policies: dict[agent_id] -> np.array (iteraciones x acciones)
    - nash_equilibrium: dict[agent_id] -> lista con probabilidad de cada acción (ej: [0.5, 0.5, ...])
    - title_suffix: string adicional para el título
    """

    agents = list(policies.keys())
    num_agents = len(agents)

    fig, axs = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5), sharey=True)
    if num_agents == 1:
        axs = [axs]

    # Definir la paleta de colores "tab20"
    # Calculamos el máximo número de acciones entre todos los agentes
    num_total_actions = max(policy.shape[1] for policy in policies.values())
    colors = sns.color_palette("tab20", n_colors=num_total_actions)

    for idx, agent in enumerate(agents):
        policy_matrix = policies[agent]
        nash = np.array(nash_equilibrium[agent])
        diffs = np.abs(policy_matrix - nash)  # shape: (T, A)

        ax = axs[idx]
        for action in range(diffs.shape[1]):
            ax.plot(
                range(len(diffs)),
                diffs[:, action],
                label=f'Action {action + 1}',
                linewidth=2,
                color=colors[action % num_total_actions]
            )
        ax.set_title(f'Convergence for {agent} {title_suffix}')
        ax.set_xlabel("Iteration")
        if idx == 0:
            ax.set_ylabel("Abs. Error per Action")
        ax.legend()
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




def plot_dual_agent_simplex(
    empirical_distributions: dict,
    action_labels: list = None,
    action1_idx: int = 0,
    action2_idx: int = 1,
    title="Empirical distributions in dual simplex"
):
    assert len(empirical_distributions) == 2, "Debe haber exactamente dos agentes."
    assert len(action_labels) == 3, "action_labels debe contener los nombres de las 3 acciones (ej. ['Rock', 'Paper', 'Scissors'])."
    assert 0 <= action1_idx < 3 and 0 <= action2_idx < 3 and action1_idx != action2_idx, \
        "Los índices de acción deben ser 0, 1 o 2 y diferentes entre sí."

    agent_names = list(empirical_distributions.keys())
    colors = ['steelblue', 'darkorange']
    n_points = len(next(iter(empirical_distributions.values())))

    # Determine the third action index
    all_indices = {0, 1, 2}
    remaining_idx = list(all_indices - {action1_idx, action2_idx})[0]

    plt.figure(figsize=(7, 7))

    # Diagonal compartida (π₃ = 0, where π₃ is the probability of the *remaining* action)
    plt.plot([0, 1], [1, 0], 'k--', linewidth=1.2, label=f"$\pi_3$ ({action_labels[remaining_idx]}) = 0")

    for i, agent in enumerate(agent_names):
        traj = np.array(empirical_distributions[agent])

        # Extract the probabilities for the chosen actions
        prob_action1 = traj[:, action1_idx]
        prob_action2 = traj[:, action2_idx]
        prob_remaining = 1 - prob_action1 - prob_action2

        valid = (prob_action1 >= 0) & (prob_action2 >= 0) & (prob_remaining >= 0)
        prob_action1, prob_action2 = prob_action1[valid], prob_action2[valid]

        if i == 1:
            plot_x = 1 - prob_action2
            plot_y = 1 - prob_action1
        else:
            plot_x = prob_action1
            plot_y = prob_action2

        # Gradient of alpha
        alphas = np.linspace(0.2, 1.0, len(plot_x))
        for j in range(1, len(plot_x)):
            plt.plot(
                [plot_x[j - 1], plot_x[j]], [plot_y[j - 1], plot_y[j]],
                color=colors[i], alpha=alphas[j], lw=2
            )

        # Discrete points every 10 steps
        plt.scatter(plot_x[::10], plot_y[::10], color=colors[i], s=15, alpha=0.6, label=f"{agent}")

        # Nash Equilibrium (1/3, 1/3, 1/3) for Rock-Paper-Scissors
        x_eq_nash = 1/3
        y_eq_nash = 1/3

        if i == 1:
            plot_x_eq = 1 - y_eq_nash
            plot_y_eq = 1 - x_eq_nash
        else:
            plot_x_eq = x_eq_nash
            plot_y_eq = y_eq_nash
        plt.plot(plot_x_eq, plot_y_eq, 'k*', markersize=14)


    # Axes and style
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ticks = np.linspace(0, 1, 6)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')

    # Primary axes (agent 1) - CORRECCIÓN APLICADA AQUÍ
    plt.xlabel(r"$\pi_1(\mathrm{" + action_labels[action1_idx] + "})$", loc='center')
    plt.ylabel(r"$\pi_1(\mathrm{" + action_labels[action2_idx] + "})$", loc='center')

    # Secondary axes (agent 2) - CORRECCIÓN APLICADA AQUÍ
    ax2 = plt.gca().secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))
    ax2.set_xticks(ticks)
    ax2.set_xlabel(r"$\pi_2(\mathrm{" + action_labels[action1_idx] + "})$", loc='center')

    ax3 = plt.gca().secondary_yaxis('right', functions=(lambda y: 1 - y, lambda y: 1 - y))
    ax3.set_yticks(ticks)
    ax3.set_ylabel(r"$\pi_2(\mathrm{" + action_labels[action2_idx] + "})$", loc='center')

    # Legend and title
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

