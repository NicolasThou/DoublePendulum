from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from domain import Domain
import utils


# discretization of the action space
action_space = [-1, -0.5, -0.25, 0.25, 0.5, 1]


def NeuralNetwork():
    net = Sequential([
        Dense(10, input_shape=(10,), activation='relu'),
        Dropout(0.4),
        Dense(5, activation='relu'),
        Dropout(0.4),
        Dense(1)
    ])

    sgd = optimizers.SGD(learning_rate=0.01)
    net.compile(loss='MSE', optimizer=sgd, metrics=['mse'])

    return net


def delta(step, model):
    """
    Compute the temporal difference
    """
    if model is None:
        td = step[2]
    else:
        predictions = []
        for u in action_space:
            x = np.concatenate((step[3], [u]))
            x = np.array([x])
            p = model.predict(x).item()
            predictions.append(p)

        max_prediction = np.max(predictions)
        x = np.concatenate((step[0], step[1]))
        x = np.array([x])
        td = step[2] + utils.gamma * max_prediction - model.predict(x).item()

    return td


def build_training_set_parametric_Q_Learning(F, model_build):
    """
    Build the training set for training the parametric
    approximation architecture for the Q-Learning

    """
    inputs = []  # input set
    outputs = []  # output set
    for step in F:
        i = np.concatenate((step[0], step[1]))

        if step[4]:
            # case where the state t+1 is a final state
            o = step[2]
        else:
            o = delta(step[:-1], model_build)  # TODO : is this really Parametric Q-Learning ?!

        # add the new sample in the training set
        inputs.append(i)
        outputs.append(o)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


def td_error(model):
    """
    Computes an estimation of TD-error for a model
    """
    d = Domain()
    s = d.initial_state()
    deltas = []

    # expectation over 20 trials
    for i in range(20):
        # sample random action to computes a four-tuple
        u = d.random_action()
        next_s, r = d.f(u)

        td = delta([s, u, r, next_s], model)
        deltas.append(td)

        if d.is_final_state():
            s = d.initial_state()
        else:
            s = next_s

    return np.mean(deltas)


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

    # Initialize the model
    model_Q_learning = NeuralNetwork()

    for k in range(N):
        print("=================== iteration k = {} ======================".format(k))

        # build batch trajectory with 500 random samples of F
        idx = np.random.choice(range(len(F)), size=500).tolist()
        f = np.array(F)[idx].tolist()

        if k == 0:
            X, y = build_training_set_parametric_Q_Learning(f, None)
        else:
            X, y = build_training_set_parametric_Q_Learning(f, model_Q_learning)

        loss = model_Q_learning.fit(X, y, batch_size=32, epochs=50, verbose=0)
        loss = loss.history['loss']

        # computes for each iteration the delta with the updated model
        d = td_error(model_Q_learning)
        print(f'delta = {d}')
        temporal_difference.append(d)

    return model_Q_learning, temporal_difference


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
    model, delta_loss = Q_learning_parametric_function(trajectory, 50)

    plt.plot(range(len(delta_loss)), delta_loss)
    plt.show()
