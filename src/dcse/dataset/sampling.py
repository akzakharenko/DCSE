import random

def sample_parameters(param_space, min_pairs, max_pairs, total_species):
    feasible_combos = []

    for N in param_space["target_count"]:
        for K in param_space["K"]:
            if K >= N:
                continue
            if (N * K) % 2 != 0:
                continue

            total_pairs = N * K // 2
            if min_pairs <= total_pairs <= max_pairs:
                feasible_combos.append((N, K))

    if not feasible_combos:
        raise ValueError("No feasible (target_count, K) combinations")

    N, K = random.choice(feasible_combos)

    return {
        "target_count": N,
        "K": K,
        "num_species": random.randint(1, total_species),
        "random_seed": random.randint(0, 1_000_000),
    }