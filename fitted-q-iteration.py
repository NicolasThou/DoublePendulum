from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from matplotlib import pyplot as plt
import utils
from domain import Domain


# discretization of the action space
action_space = [-1, -0.5, -0.25, 0.25, 0.5, 1]


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


class MeanPolicy:
    """
    Action space is continuous
    """
    def __init__(self, Q, action_space, temperature=1.0):
        # The Q-function have been trained on the discretized action space
        self.Q = Q

        # The action space is discrete
        self.action_space = action_space
        self.temperature = temperature

    def __call__(self, x):
        """
        We weight the actions by their value of the Q-function on this state
        """
        weights = []
        for u in self.action_space:
            input = np.concatenate((x, [u]))
            weights.append(self.Q.predict([input]).item())

        # normalize so the sum equal 1
        weights = sigmoid(weights, self.temperature)

        # take the weighted average of actions
        action = np.average(self.action_space, weights=weights)
        return action


def sigmoid(l, t=1.0):
    l = np.array(l)
    denominator = np.exp(-l/t)
    denominator = np.sum(denominator)
    return (np.exp(-l/t)/denominator).tolist()


def discretize_space(space_min, space_max, nb_values):
    space = []
    cst = (space_max - space_min) / (nb_values - 1)
    for k in range(nb_values - 1):
        space.append(space_min + k * cst)
    space.append(space_max)
    return space


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
    loss = []

    # Q_0 = 0 everywhere
    Q = ExtraTreesRegressor()
    Q.fit(X=[[0 for i in range(10)]], y=[0])
    Q_list.append(Q)

    for n in range(N):
        print(f'n = {n+1}')
        X, y = build_training_set(Q_list[-1], F)
        Q = ExtraTreesRegressor(n_estimators=M, min_samples_split=n_min, oob_score=True, bootstrap=True)
        Q.fit(X, y)
        Q_list.append(Q)
        loss.append(Q.oob_score_)

    return Q_list, loss


if __name__ == '__main__':
    F = utils.build_trajectory(1000)

    Q_list, loss = fitted_q_iteration(F, N=50)
    print('Fitted-Q-Iteration over')

    # J = []
    # for i in range(50):
    #     print(i)
    #
    #     Q = Q_list[i]
    #     j_list = []
    #     mu = Policy(Q, action_space=[-1, 1])
    #     for n_samples in range(5):
    #         j_list.append(utils.J(mu, 20))
    #
    #     J.append(np.mean(j_list))

    plt.plot(range(len(loss)), loss)
    plt.show()

    # d = Domain()
    # s = d.initial_state()
    # policy = MeanPolicy(Q_list[-1], action_space, 0.05)
    # action = policy(s)
    # print(action)
