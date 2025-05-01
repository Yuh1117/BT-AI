# cau 1
states = ["S0", "S1", "S2"]
actions = ["a0", "a1"]

transition_probs = {
    "S0": {
        "a0": {"S0": 0.5, "S1": 0.0, "S2": 0.5},
        'a1': {'S0': 0.0, 'S1': 0.0, 'S2': 1.0}
    },
    "S1": {
        "a0": {"S0": 0.7, "S1": 0.1, "S2": 0.2},
        'a1': {'S0': 0.0, 'S1': 0.95, 'S2': 0.05}
    },
    "S2": {
        "a0": {"S0": 0.4, "S1": 0.0, "S2": 0.6},
        'a1': {'S0': 0.3, 'S1': 0.3, 'S2': 0.4}
    },
}


def reward(s, a, s_prime):
    if a == 'a0':
        return 0
    else:
        return -1


print("Tập trạng thái: ", states)
print("Tập hành động: ", actions)
print("\nHàm chuyển tiếp T(s, a, s'):")
for s in states:
    for a in actions:
        print(f"Từ {s}, hành động {a}:")
        for s_prime in states:
            prob = transition_probs[s][a][s_prime]
            print(f" -> {s_prime}: {prob}")

print("\nHàm phần thưởng R(s, a, s'):")
for s in states:
    for a in actions:
        for s_prime in states:
            r = reward(s, a, s_prime)
            print(f"R({s}, {a}, {s_prime}) = {r}")

# cau 2
# Tham số Value Iteration
print("Tập trạng thái: ", states)
print("Tập hành động: ", actions)
print("\nHàm chuyển tiếp T(s, a, s'):")
for s in states:
    for a in actions:
        print(f"Từ {s}, hành động {a}:")
        for s_prime in states:
            prob = transition_probs[s][a][s_prime]
            print(f" -> {s_prime}: {prob}")

print("\nHàm phần thưởng R(s, a, s'):")
for s in states:
    for a in actions:
        for s_prime in states:
            r = reward(s, a, s_prime)
            print(f"R({s}, {a}, {s_prime}) = {r}")

GAMMA = 0.9
NUM_ITERATIONS = 10


def value_iteration():
    V = {s: 0 for s in states}
    policy = {s: None for s in states}

    for iteration in range(NUM_ITERATIONS):
        V_new = V.copy()
        for s in states:
            max_value = float("-inf")
            best_action = None
            for a in actions:
                value = 0
                for s_prime in states:
                    prob = transition_probs[s][a][s_prime]
                    value += prob * (reward(s, a, s_prime) + GAMMA * V[s_prime])
                if value > max_value:
                    max_value = value
                    best_action = a
            V_new[s] = max_value
            policy[s] = best_action
        V = V_new

        print(f"\nLặp {iteration + 1}:")
        for s in states:
            print(f"V({s}) = {V[s]:.4f}")

    return V, policy


V, policy = value_iteration()
print("\nChính sách tối ưu: ")
for s in states:
    print(f"π({s}) = {policy[s]}")

# cau 3
states = ["S0", "S1", "S2"]
actions = ["di_chuyen", "don_dep"]

transition_probs = {
    "S0": {
        "di_chuyen": {"S0": 0.5, "S1": 0.0, "S2": 0.5},
        'don_dep': {'S0': 0.0, 'S1': 0.0, 'S2': 1.0}
    },
    "S1": {
        "di_chuyen": {"S0": 0.7, "S1": 0.1, "S2": 0.2},
        'don_dep': {'S0': 0.0, 'S1': 0.95, 'S2': 0.05}
    },
    "S2": {
        "di_chuyen": {"S0": 0.4, "S1": 0.0, "S2": 0.6},
        'don_dep': {'S0': 0.3, 'S1': 0.3, 'S2': 0.4}
    },
}


def reward(s, a, s_prime):
    if a == 'di_chuyen':
        return 0
    else:
        if s == "S1":
            return 2
        else:
            return -1


print("Tập trạng thái: ", states)
print("Tập hành động: ", actions)
print("\nHàm chuyển tiếp T(s, a, s'):")
for s in states:
    for a in actions:
        print(f"Từ {s}, hành động {a}:")
        for s_prime in states:
            prob = transition_probs[s][a][s_prime]
            print(f" -> {s_prime}: {prob}")

print("\nHàm phần thưởng R(s, a, s'):")
for s in states:
    for a in actions:
        for s_prime in states:
            r = reward(s, a, s_prime)
            print(f"R({s}, {a}, {s_prime}) = {r}")

# Tham số Value Iteration
GAMMA = 0.9
NUM_ITERATIONS = 10


def value_iteration():
    V = {s: 0 for s in states}
    policy = {s: None for s in states}

    for iteration in range(NUM_ITERATIONS):
        V_new = V.copy()
        for s in states:
            max_value = float("-inf")
            best_action = None
            for a in actions:
                value = 0
                for s_prime in states:
                    prob = transition_probs[s][a][s_prime]
                    value += prob * (reward(s, a, s_prime) + GAMMA * V[s_prime])
                if value > max_value:
                    max_value = value
                    best_action = a
            V_new[s] = max_value
            policy[s] = best_action
        V = V_new

        print(f"\nLặp {iteration + 1}:")
        for s in states:
            print(f"V({s}) = {V[s]:.4f}")

    return V, policy


V, policy = value_iteration()
print("\nChính sách tối ưu: ")
for s in states:
    print(f"π({s}) = {policy[s]}")
