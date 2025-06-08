# Projet : Simulation de la propagation d'un virus (SIR) sur différents types de graphes

**Date** : 27 Mai 2025

**Objectif** : Simuler la dynamique d'une épidémie en utilisant le modèle SIR (Susceptible-Infected-Recovered) sur des graphes Erdős-Rényi (ER), Watts-Strogatz (WS), et Barabási-Albert (BA), puis comparer la vitesse et l'étendue de la propagation selon la structure du réseau.

**Auteur** : \[Votre Nom\]

## Prérequis : Installation des packages

Pour exécuter ce projet, vous devez installer les bibliothèques Python suivantes. Exécutez les commandes suivantes dans votre terminal ou invite de commande :

```bash
pip install networkx
pip install numpy
pip install matplotlib
```

Assurez-vous d'avoir Python 3.x installé. Ces packages sont nécessaires pour la génération des graphes, les calculs numériques et les visualisations.

## Contexte pédagogique

Ce projet vise à :

- Mettre en pratique les modèles de graphes (ER, WS, BA).
- Comprendre la diffusion dans des réseaux complexes.
- Travailler avec des simulations stochastiques.
- Observer des propriétés comme les hubs ou les petits mondes.

## Structure du document

1. Importation des bibliothèques
2. Implémentation des fonctions SIR
3. Simulation sur les différents graphes
4. Visualisation et analyse comparative
5. Conclusions

## 1. Importation des bibliothèques

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import choice, random

# Set random seed for reproducibility
np.random.seed(42)
```

## 2. Implémentation des fonctions SIR

### Étape 1 : Initialisation de l'état des nœuds

```python
def initialize_sir_states(G, initial_infected=1):
    """Initialize node states: all S except a few random infected nodes.
    
    Args:
        G (networkx.Graph): Input graph
        initial_infected (int): Number of initially infected nodes
    
    Returns:
        dict: Dictionary mapping nodes to their state ('S', 'I', or 'R')
    """
    states = {node: 'S' for node in G.nodes()}
    infected_nodes = np.random.choice(list(G.nodes()), size=initial_infected, replace=False)
    for node in infected_nodes:
        states[node] = 'I'
    return states
```

### Étape 2 : Une étape de simulation SIR

```python
def sir_step(G, states, beta, gamma):
    """Apply one step of the SIR simulation and return updated states.
    
    Args:
        G (networkx.Graph): Input graph
        states (dict): Current node states
        beta (float): Infection probability
        gamma (float): Recovery probability
    
    Returns:
        dict: Updated node states
        tuple: (S_count, I_count, R_count) for current step
    """
    new_states = states.copy()
    
    # Infection step
    for node in G.nodes():
        if states[node] == 'I':
            for neighbor in G.neighbors(node):
                if states[neighbor] == 'S' and random() < beta:
                    new_states[neighbor] = 'I'
    
    # Recovery step
    for node in G.nodes():
        if states[node] == 'I' and random() < gamma:
            new_states[node] = 'R'
    
    # Count states
    S_count = sum(1 for state in new_states.values() if state == 'S')
    I_count = sum(1 for state in new_states.values() if state == 'I')
    R_count = sum(1 for state in new_states.values() if state == 'R')
    
    return new_states, (S_count, I_count, R_count)
```

### Étape 3 : Boucle de simulation complète

```python
def run_sir_simulation(G, beta=0.03, gamma=0.01, initial_infected=1):
    """Run full SIR simulation until no infected nodes remain.
    
    Args:
        G (networkx.Graph): Input graph
        beta (float): Infection probability
        gamma (float): Recovery probability
        initial_infected (int): Number of initially infected nodes
    
    Returns:
        list: List of (S_count, I_count, R_count) tuples for each time step
    """
    states = initialize_sir_states(G, initial_infected)
    results = []
    
    while sum(1 for state in states.values() if state == 'I') > 0:
        states, counts = sir_step(G, states, beta, gamma)
        results.append(counts)
    
    return results
```

### Étape 4 : Comparaison des structures

```python
def compare_structures(n, p, k, beta_ws, m, beta_sir, gamma_sir):
    """Run SIR simulation on ER, WS, and BA graphs and return results.
    
    Args:
        n (int): Number of nodes
        p (float): Edge probability for ER graph
        k (int): Number of nearest neighbors for WS graph
        beta_ws (float): Rewiring probability for WS graph
        m (int): Number of edges to attach for BA graph
        beta_sir (float): Infection probability for SIR
        gamma_sir (float): Recovery probability for SIR
    
    Returns:
        dict: Dictionary with graph type as key and SIR results as value
    """
    # Create graphs
    G_er = nx.erdos_renyi_graph(n, p)
    G_ws = nx.watts_strogatz_graph(n, k, beta_ws)
    G_ba = nx.barabasi_albert_graph(n, m)
    
    # Run simulations
    results = {
        'ER': run_sir_simulation(G_er, beta_sir, gamma_sir),
        'WS': run_sir_simulation(G_ws, beta_sir, gamma_sir),
        'BA': run_sir_simulation(G_ba, beta_sir, gamma_sir)
    }
    
    return results
```

### Étape 5 : Affichage et analyse

```python
def plot_sir_dynamics(sir_results, title="SIR Dynamics"):
    """Plot S, I, R curves over time.
    
    Args:
        sir_results (dict): Dictionary with graph type and SIR results
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for graph_type, results in sir_results.items():
        S = [r[0] for r in results]
        I = [r[1] for r in results]
        R = [r[2] for r in results]
        time = range(len(results))
        
        plt.plot(time, S, label=f'{graph_type} - Susceptible', linestyle='--')
        plt.plot(time, I, label=f'{graph_type} - Infected', linestyle='-')
        plt.plot(time, R, label=f'{graph_type} - Recovered', linestyle='-.')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Nodes')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 3. Simulation et Analyse Comparative

Nous exécutons la simulation avec les paramètres suivants :

- Nombre de nœuds : 1000
- Probabilité de connexion (ER) : p=0.01
- Voisins les plus proches (WS) : k=4
- Probabilité de recâblage (WS) : beta_ws=0.1
- Arêtes par nœud (BA) : m=2
- Probabilité d'infection : beta_sir=0.03
- Probabilité de guérison : gamma_sir=0.01

```python
# Run simulation
n = 1000
p = 0.01
k = 4
beta_ws = 0.1
m = 2
beta_sir = 0.03
gamma_sir = 0.01

results = compare_structures(n, p, k, beta_ws, m, beta_sir, gamma_sir)
plot_sir_dynamics(results, title="SIR Dynamics on Different Graph Structures")
```

**Note** : L'exécution du code ci-dessus générera un graphique montrant l'évolution des populations S, I, et R pour chaque type de graphe. Comme ce document est en Markdown, le graphique ne peut pas être affiché directement ici, mais il sera visible lors de l'exécution dans un environnement Python.

## 4. Analyse Comparative

### Vitesse d'infection

- **Graphe ER** : La propagation est modérée, dépendant fortement de la probabilité de connexion $p$. Avec $p=0.01$, le graphe est relativement peu connecté, ce qui ralentit la propagation.
- **Graphe WS** : La propagation est plus rapide en raison de la structure en petits mondes, surtout avec un faible $\\beta\_{ws}$ qui maintient des clusters locaux favorisant la transmission rapide au sein des groupes.
- **Graphe BA** : La propagation est la plus rapide, surtout si les hubs (nœuds à forte connectivité) sont infectés tôt, car ils agissent comme des super-propagateurs.

### Pourcentage total infecté

Pour quantifier le pourcentage total infecté, nous calculons la proportion de nœuds ayant atteint l'état 'R' à la fin de la simulation.

```python
def calculate_total_infected(results):
    """Calculate the total percentage of infected nodes for each graph.
    
    Args:
        results (dict): SIR simulation results
    
    Returns:
        dict: Percentage of total infected nodes per graph type
    """
    total_infected = {}
    for graph_type, data in results.items():
        final_R = data[-1][2] if data else 0
        total_infected[graph_type] = (final_R / n) * 100
    return total_infected

total_infected = calculate_total_infected(results)
for graph_type, percentage in total_infected.items():
    print(f'{graph_type}: {percentage:.2f}% nodes were infected')
```

**Note** : Les résultats exacts dépendent de l'exécution stochastique, mais en général, le graphe BA montre un pourcentage plus élevé d'infectés en raison de ses hubs, suivi par WS, puis ER.

### Rôle de la structure du réseau

- **ER** : La structure aléatoire entraîne une propagation homogène mais lente, car il n'y a pas de hubs pour accélérer la diffusion.
- **WS** : La propriété de petit monde permet une propagation rapide au sein des clusters locaux, mais la portée globale dépend du recâblage.
- **BA** : Les hubs jouent un rôle crucial, permettant une propagation explosive si infectés tôt, mais la résilience du réseau peut être faible si les hubs sont immunisés.

## 5. Conclusions

La simulation montre que la structure du réseau a un impact significatif sur la dynamique épidémique :

- Le graphe BA est le plus vulnérable à une propagation rapide en raison de ses hubs.
- Le graphe WS favorise une propagation rapide dans des communautés locales.
- Le graphe ER montre une propagation plus lente et plus uniforme.

Ces résultats soulignent l'importance de la topologie du réseau dans la gestion des épidémies, avec des implications pour la modélisation de la diffusion d'informations ou de malwares.