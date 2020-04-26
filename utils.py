import numpy as np
from matplotlib import pyplot as plt
from domain import Domain

gamma = 0.9


class Policy:
    """
    Action space is discrete
    """
    def __init__(self, Q, action_space):
        self.Q = Q
        self.action_space = action_space

    def __call__(self, x):
        values = []
        for u in self.action_space:
            input = np.concatenate((x, [u]))
            values.append(self.Q.predict([input]))

        max_index = np.argmax(values).item()
        return [self.action_space[max_index]]


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


def build_trajectory(size):
    F = []
    d = Domain()
    x = d.env.reset()
    for i in range(size):
        u = d.random_action()
        new_x, r = d.f(u)
        F.append([x, u, r, new_x])
        if d.is_final_state():
            x = d.env.reset()
        else:
            x = new_x

    return F


if __name__ == '__main__':
    # policy return rndomly -1 or 1
    mu = lambda x: np.random.choice([-1, 1], 1).tolist()

    # list of expected return

    j = []
    for N in range(100):
        j.append(J(mu, 50))

    # display expected return
    plt.plot(range(100), j)
    plt.show()
