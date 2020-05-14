from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump, load
import numpy as np
from matplotlib import pyplot as plt
import utils
from domain import Domain


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

        # normalize so the sum equal 1 = sigmoid
        denominator = np.exp(-np.array(weights)/self.temperature)
        denominator = np.sum(denominator)
        weights = (np.exp(-np.array(weights)/self.temperature)/denominator).tolist()

        # take the weighted average of actions
        action = np.average(self.action_space, weights=weights)
        return action


def build_training_set(Q, F, action_space):
    """
    Build the training set using the Q-function and the trajectory F
    """
    # training set
    X = []
    y = []

    for x, u, r, next_x, is_final_state in F:

        if not is_final_state:
            # extracting maxQ(x',u')
            predictions = []
            for action in action_space:
                input = np.concatenate((next_x, [action]))
                predictions.append(Q.predict([input]))
            y.append(r + utils.gamma * np.max(predictions))
        else:
            y.append(r)

        X.append(np.concatenate((x, u)))

    return X, y


def fitted_q_iteration(F, action_space, N=100, n_min=2, M=50):
    """
    Apply the Fitted-Q-Iteration algorithm with Extra-Tree
    (discrete action space !)
        see 'Tree-Based Batch Mode Reinforcement Learning' D. ERnts, P. Geurts and L. Wehenkel; p.34
    """
    Q_list = []
    loss = []
    td_loss = []

    # Q_0 = 0 everywhere
    Q = ExtraTreesRegressor()
    Q.fit(X=[[0 for i in range(10)]], y=[0])
    Q_list.append(Q)

    for n in range(N):
        print(f'n = {n+1}')
        X, y = build_training_set(Q_list[-1], F, action_space)
        Q = ExtraTreesRegressor(n_estimators=M, min_samples_split=n_min, oob_score=True, bootstrap=True)
        Q.fit(X, y)
        Q_list.append(Q)
        loss.append(Q.oob_score_)
        td_loss.append(utils.td_error(Q, action_space))
        print(f'loss : {loss[-1]} | td error : {td_loss[-1]}')

    return Q_list, loss, td_loss


def train_ExtraTree(action_space):
    F = utils.build_trajectory(10000, from_action_space=True, action_space=action_space)
    Q_list, loss, td_loss = fitted_q_iteration(F, action_space, N=50)
    return Q_list[-1]


if __name__ == '__main__':
    # load a model and create a policy from it
    # !!! the action space used has to match the action space used for the training !!!
    model = load('models/tree 6-dimensional.joblib')
    mu = Policy(model, action_space=[-1, -0.5, -0.25, 0.25, 0.5, 1])

    j_list = []
    for i in range(1000):  # compute 1000 episode to have the expected return as an average
        j_list.append(utils.J(mu, 100))

    print(f'Expected return of Extra-Tree : {np.mean(j_list)}')
