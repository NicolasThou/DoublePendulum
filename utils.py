import numpy as np
from matplotlib import pyplot as plt
from domain import Domain

gamma = 0.95


def J(policy, N, d=None, x=None):
    """
    Compute the expected return of a policy for a state
    """
    if N == 0:
        return 0
    else:
        if d is not None:
            u = policy(x)
            new_x, r = d.f(u)
            return r + gamma*J(policy, N-1, d, new_x)
        else:
            # else we create it
            d = Domain()
            x = d.initial_state()
            u = policy(x)
            new_x, r = d.f(u)
            return r + gamma*J(policy, N-1, d, new_x)


def build_trajectory(size, from_action_space=False, action_space=None):
    T= []
    d = Domain()
    x = d.initial_state()
    while len(T) < size:
        if not from_action_space:
            u = d.random_action()
        else:
            # choose an action from discrete action space
            u = np.random.choice(action_space, size=1)

        new_x, r = d.f(u)
        T.append([x, u, r, new_x, d.is_final_state()])

        if d.is_final_state():
            x = d.initial_state()
        else:
            x = new_x

    np.random.shuffle(T)
    return T


if __name__ == '__main__':
    # policy return randomly -1 or 1
    mu = lambda x: np.random.choice([-1, 1], 1).tolist()

    # list of expected return

    j = []
    for N in range(100):
        j.append(J(mu, 50))

    # display expected return
    plt.plot(range(100), j)
    plt.show()
