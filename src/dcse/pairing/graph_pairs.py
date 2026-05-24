import networkx as nx


def generate_pairs(records, k: int, seed: int = 42):
    N = len(records)

    if k >= N:
        raise ValueError("K must be less than N")

    if (N * k) % 2 != 0:
        raise ValueError("N*K must be even")

    G = nx.random_regular_graph(d=k, n=N, seed=seed)

    id_list = [r.id for r in records]

    pairs = []
    for u, v in G.edges():
        pairs.append((id_list[u], id_list[v]))

    return pairs