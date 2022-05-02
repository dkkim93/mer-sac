import pickle
import numpy as np
from tarjan import tarjan
from scipy.io import savemat


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def get_j_action(agents, state):
    j_action = agents[0].actor[state]
    (alpha, beta) = agents[1].actor[state]
    j_action += "A" if alpha >= beta else "B"
    return j_action


def get_current_transition(agents, transition):
    current_transition = {}

    for state in transition.keys():
        i_action = agents[0].actor[state]
        next_state = transition[state][i_action]
        current_transition[state] = next_state

    return current_transition


def save_transition(agents, transition, args, name):
    size = int(len(transition))
    P = np.zeros((size, size))  # Initialize
    for state in transition.keys():
        next_states, next_probs = transition[state]
        P[state, next_states[0]] = next_probs[0]
        P[state, next_states[1]] = next_probs[1]
    filename = "log/tb_" + args.log_name + "/" + name
    savemat(filename, {"P": P})


def get_tarjan(transition):
    filtered_transition = {}
    for state in transition.keys():
        next_states, next_probs = transition[state]
        filtered_next_states = []
        for next_state, next_prob in zip(next_states, next_probs):
            if next_prob > 0:
                filtered_next_states.append(next_state)
        filtered_transition[state] = filtered_next_states
    return tarjan(filtered_transition)


def get_recurrent_classes(transition, agents):
    recurrent_classes = []

    # Identify recurrent states with period of 1
    for state in transition.keys():
        next_states, next_probs = transition[state]
        index = np.where(np.round(next_probs, decimals=5) == 1.)[0]
        if len(index) > 0:
            assert len(index) == 1
            if state == next_states[index[0]]:
                recurrent_classes.append([state])

    # Identify recurrent states with period > 1
    max_period = 1
    for subchain in get_tarjan(transition):
        if len(subchain) > 1:
            # Check whether recurrent classes repeat themselves
            next_states_ = []
            for state in subchain:
                next_states, next_probs = transition[state]
                for next_state, next_prob in zip(next_states, next_probs):
                    if next_prob > 0:
                        next_states_.append(next_state)

            if set(next_states_) == set(subchain):
                recurrent_classes.append(subchain)
                if len(subchain) > max_period:
                    max_period = len(subchain)

    return recurrent_classes, max_period


def policy_evaluation(env, transition, agents, recurrent_classes, args):
    if args.is_average_obj:
        return average_reward_policy_evaluation(env, transition, agents, recurrent_classes, args)
    else:
        return discounted_reward_policy_evaluation(env, transition, agents, recurrent_classes, args)


def average_reward_policy_evaluation(env, transition, agents, recurrent_classes, args):
    gain = [None for _ in transition.keys()]
    bias = [None for _ in transition.keys()]

    # Apply condition 9.2.3: Set zero bias for each
    # recurrent class's minimal index
    for recurrent_class in recurrent_classes:
        minimal_state = np.min(recurrent_class)
        bias[minimal_state] = 0.

    # Construct variables
    variables, index = {}, 0
    for state in transition.keys():
        variables["g" + str(state)] = index
        index += 1

    for state in transition.keys():
        if bias[state] is None:
            variables["b" + str(state)] = index
            index += 1

    # Get linear equations: Ax = b, where x is unknown variables
    A, b = [], []
    for state in transition.keys():
        # Get reward
        i_action = agents[0].actor[state]
        alpha, beta = agents[1].actor[state]
        j_action = "A" if alpha >= beta else "B"
        actions = i_action + j_action
        reward = (1. - args.epsilon_exploration) * env[actions][0]

        # Add exploration reward
        j_action = "B" if alpha >= beta else "A"
        actions = i_action + j_action
        reward += args.epsilon_exploration * env[actions][0]

        # Get transition
        next_states, next_probs = transition[state]

        # Add gain eqn
        a = np.zeros((len(variables),))
        a[variables["g" + str(state)]] += 1
        for next_state, next_prob in zip(next_states, next_probs):
            a[variables["g" + str(next_state)]] += -next_prob
        A.append(a)
        b.append(0)

        # Add bias eqn
        a = np.zeros((len(variables),))
        a[variables["g" + str(state)]] = 1
        if bias[state] is None:
            a[variables["b" + str(state)]] += 1
        for next_state, next_prob in zip(next_states, next_probs):
            if bias[next_state] is None:
                a[variables["b" + str(next_state)]] += -next_prob
        A.append(a)
        b.append(reward)
    A = np.stack(A)
    b = np.stack(b)
    assert A.shape[0] >= len(variables), "Less number of equations than number of unknown variables"

    # Solve linear equations
    solution, error, _, _ = np.linalg.lstsq(A, b, rcond=None)
    if not isinstance(error, np.ndarray):
        assert error <= 1e-5, "Error is too large, resulting in incorrect policy evaluation"

    for key, value in zip(variables.keys(), solution):
        state = int(str(key)[1:])
        if str(key)[0] == "g":
            gain[state] = float(value)
        else:
            bias[state] = float(value)

    return gain, bias


def discounted_reward_policy_evaluation(env, transition, agents, recurrent_classes, args):
    for state in transition.keys():
        # Get reward
        i_action = agents[0].actor[state]
        alpha, beta = agents[1].actor[state]
        j_action = "A" if alpha >= beta else "B"
        actions = i_action + j_action
        reward = (1. - args.epsilon_exploration) * env[actions][0]

        # Get transition
        next_states, next_probs = transition[state]

        # Update value
        next_value = 0.
        for next_state, next_prob in zip(next_states, next_probs):
            next_value += next_prob * agents[0].value[next_state]
        target = reward + args.discount * next_value
        agents[0].value[state] = target

    return list(agents[0].value.values()), 0.


def policy_improvement(env, gain, bias, agents, z_transition, args):
    if args.is_average_obj:
        return average_reward_policy_improvement(env, gain, bias, agents, z_transition, args)
    else:
        return discounted_reward_policy_improvement(env, gain, bias, agents, z_transition, args)


def average_reward_policy_improvement(env, gain, bias, agents, z_transition, args):
    policy = {}

    for state in z_transition.keys():
        values = []
        for i_action in ["A", "B"]:
            next_states, next_probs = z_transition[state][i_action]
            value = 0
            for next_state, next_prob in zip(next_states, next_probs):
                value += next_prob * gain[next_state]
            values.append(np.round(value, decimals=3))

        if values[0] == values[1]:
            # Applying condition B if no gain improvement can be made
            values = []
            for i_action in ["A", "B"]:
                alpha, beta = agents[1].actor[state]

                # Get reward
                alpha, beta = agents[1].actor[state]
                j_action = "A" if alpha >= beta else "B"
                actions = i_action + j_action
                reward = (1. - args.epsilon_exploration) * env[actions][0]

                # Add exploration reward
                j_action = "B" if alpha >= beta else "A"
                actions = i_action + j_action
                reward += args.epsilon_exploration * env[actions][0]

                next_states, next_probs = z_transition[state][i_action]
                value = reward
                for next_state, next_prob in zip(next_states, next_probs):
                    value += next_prob * bias[next_state]
                values.append(np.round(value, decimals=3))

            if values[0] == values[1]:
                policy[state] = agents[0].actor[state]
            else:
                if np.argmax(values) == 0:
                    policy[state] = "A"
                else:
                    policy[state] = "B"
        else:
            if np.argmax(values) == 0:
                policy[state] = "A"
            else:
                policy[state] = "B"

    return policy


def discounted_reward_policy_improvement(env, gain, bias, agents, z_transition, args):
    policy = {}

    for state in z_transition.keys():
        values = []
        for i_action in ["A", "B"]:
            alpha, beta = agents[1].actor[state]

            # Get reward
            alpha, beta = agents[1].actor[state]
            j_action = "A" if alpha >= beta else "B"
            actions = i_action + j_action
            reward = env[actions][0]

            next_states, next_probs = z_transition[state][i_action]
            next_value = 0.
            for next_state, next_prob in zip(next_states, next_probs):
                next_value += next_prob * gain[next_state]
            values.append(np.round(reward + args.discount * next_value, decimals=3))

        if np.argmax(values) == 0:
            policy[state] = "A"
        else:
            policy[state] = "B"

    return policy


def check_same_policy(policy, new_policy):
    n_same = 0
    for state in policy.keys():
        if policy[state] == new_policy[state]:
            n_same += 1

    if n_same == len(policy.keys()):
        return True
    else:
        return False


def find_closest(actor, j_action, z, next_z, args):
    diff = np.linalg.norm(actor - next_z, axis=1)
    sorted_diff_indices = np.argsort(diff)
    closest_diffs = diff[sorted_diff_indices[0:2]]
    closest_diff_indices = sorted_diff_indices[0:2]

    # Check whether the second closest point is
    # in the alpha or beta direction
    alpha, beta = z
    if j_action == "A":
        if actor[sorted_diff_indices[1]][1] != beta:
            corrected_index = 2
            while actor[sorted_diff_indices[corrected_index]][1] != beta:
                corrected_index += 1
            closest_diffs[1] = diff[sorted_diff_indices[corrected_index]]
            closest_diff_indices[1] = sorted_diff_indices[corrected_index]
            assert actor[closest_diff_indices[1]][1] == beta
    else:
        if actor[sorted_diff_indices[1]][0] != alpha:
            corrected_index = 2
            while actor[sorted_diff_indices[corrected_index]][0] != alpha:
                corrected_index += 1
            closest_diffs[1] = diff[sorted_diff_indices[corrected_index]]
            closest_diff_indices[1] = sorted_diff_indices[corrected_index]
            assert actor[closest_diff_indices[1]][0] == alpha

    logits = np.sum(closest_diffs) - closest_diffs
    probs = np.round(logits / np.sum(logits), decimals=3)
    assert probs[0] >= 0 and probs[0] <= 1
    assert probs[1] >= 0 and probs[1] <= 1

    # Apply perturbation
    if args.is_perturbation:
        raise NotImplementedError()

    assert np.round(np.sum(probs), decimals=3) == 1, "{}".format(probs)
    return closest_diff_indices, probs


def save_result(agents, gain, bias, transition, iteration, args):
    filename = "log/tb_" + args.log_name + "/" + "meta_agent" + str(iteration) + ".pkl"
    with open(filename, 'wb') as f:
        pickle.dump(agents[0].actor, f)

    filename = "log/tb_" + args.log_name + "/" + "opponent" + str(iteration) + ".npy"
    np.save(filename, agents[-1].actor)

    filename = "log/tb_" + args.log_name + "/" + "gain" + str(iteration) + ".npy"
    np.save(filename, gain)

    filename = "log/tb_" + args.log_name + "/" + "bias" + str(iteration) + ".npy"
    np.save(filename, bias)

    filename = "log/tb_" + args.log_name + "/" + "transition" + str(iteration) + ".pkl"
    with open(filename, 'wb') as f:
        pickle.dump(transition, f)
