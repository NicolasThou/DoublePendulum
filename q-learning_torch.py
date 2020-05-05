import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from domain import Domain
import utils


# discretization of the action space
action_space = [-1, -0.5, -0.25, 0.25, 0.5, 1]


def NeuralNetwork():
    net = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(5, 1)
    )

    return net


def Q_learning_parametric_function(F, N):
    """
    new_baseline_model() use a customize loss which return only the y_pred.
    Plus, we have the parameter sample_weight on the fit() method which multiply each y_pred by the correct
    delta corresponding to the sample. So we obtain : y_pred * delta in the output layer

    Then we use the SGD on the Loss : y_pred * delta in order to optimize all the parameters of
    the neural network.
    """
    # store the delta along the training
    temporal_difference = []

    # loss along N
    training_loss = []

    # Initialize the model
    model_Q_learning = NeuralNetwork()

    optimizer = optim.SGD(model_Q_learning.parameters(), lr=0.01)

    for k in range(N):
        print("=================== iteration k = {} ======================".format(k))

        # build batch trajectory with 500 random samples of F
        idx = np.random.choice(range(len(F)), size=500).tolist()
        f = np.array(F)[idx].tolist()

        l = []
        for s, u, r, next_s, is_final_state in f:
            max_pred = []
            for action in action_space:
                x = np.concatenate((next_s, [action]))
                x = torch.tensor(x, dtype=torch.float32)
                max_pred.append(model_Q_learning(x).detach().numpy().item())

            if is_final_state:
                y = r
            else:
                y = r + utils.gamma*max(max_pred)
            x = np.concatenate((s, u))
            v = model_Q_learning(torch.tensor(x, dtype=torch.float32))

            optimizer.zero_grad()
            print(f'y = {y} | v = {v}')
            loss = 0.5*(y - v)**2
            l.append(loss.item())
            loss.backward()
            optimizer.step()

        # computes for each iteration the delta with the updated model
        d = utils.td_error(model_Q_learning, action_space, nb_approximations=20)
        print(f'delta = {d}')
        temporal_difference.append(d)
        training_loss.append(np.mean(l))

    return model_Q_learning, training_loss, temporal_difference


# ------------------------ PLOT CURVE AND TENDANCY ---------------
def show(X, y, title, xlabel, ylabel):
    """
    Create a softer visualization of plots
    """
    # Fit
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    # Visualize
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue', linewidth=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    trajectory = utils.build_trajectory(1000)
    model, training_loss, delta_loss = Q_learning_parametric_function(trajectory, 5)

    plt.plot(range(len(training_loss)), training_loss)
