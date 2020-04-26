from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from matplotlib import pyplot as plt
import utils
from domain import Domain

# epsilon greedy policy
epsilon = 0.1

# discretization of the action space
action_space = [-1, 1]


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


def max_of_Q(Q, x):
    """
    Return the maxQ(x,u') term
    """
    values = []
    for u in action_space:
        input = np.concatenate((x, [u]))
        values.append(Q.predict([input]))

    return np.max(values)


def build_trajectory(size):
    F = []
    d = Domain()
    x = d.env.reset()
    for i in range(size):
        u = np.random.choice(action_space, 1).tolist()
        new_x, r = d.f(u)
        F.append([x, u, r, new_x])
        if d.is_final_state():
            x = d.env.reset()
        else:
            x = new_x

    return F


def build_training_set(Q, F):
    """
    Build the training set using the Q-function and the trajectory F
    """
    # training set
    X = []
    y = []

    for x, u, r, next_x in F:
        X.append(np.concatenate((x, u)))
        y.append(r + utils.gamma*max_of_Q(Q, next_x))

    return X, y


def fitted_q_iteration(F, N=100, n_min=2, M=50):
    """
    Apply the Fitted-Q-Iteration algorithm with Extra-Tree
    (discrete action space !)
        see 'Tree-Based Batch Mode Reinforcement Learning' D. ERnts, P. Geurts and L. Wehenkel; p.34
    """
    Q_list = []

    # Q_0 = 0 everywhere
    Q = ExtraTreesRegressor()
    Q.fit(X=[[0 for i in range(10)]], y=[0])
    Q_list.append(Q)

    for n in range(N):
        X, y = build_training_set(Q_list[-1], F)
        Q = ExtraTreesRegressor(n_estimators=M, min_samples_split=n_min)
        Q.fit(X, y)
        Q_list.append(Q)

    return Q_list


if __name__ == '__main__':
    F = utils.build_trajectory(10)

    Q_list = fitted_q_iteration(F, N=50)
    print('Fitted-Q-Iteration over')

    J = []
    for i in range(50):
        print(i)

        Q = Q_list[i]
        j_list = []
        mu = Policy(Q, action_space=[-1, 1])
        for n_samples in range(5):
            j_list.append(utils.J(mu, 20))

        J.append(np.mean(j_list))

    plt.plot(range(50), J)
    plt.show()
