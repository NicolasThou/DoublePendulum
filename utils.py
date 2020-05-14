import numpy as np
from domain import Domain

gamma = 0.95


class ExtraTreePolicy:
    """
    Class used to build a policy from the Extra-Tree models
    """
    def __init__(self, model, action_space):
        """
        We have to precise the action space used
        It has to be the same as the one for trainnig the model used
        """
        self.model = model
        self.action_space = action_space

    def __call__(self, x):
        """
        Return the action from action space with the highest probability
        The distributions is predicted by the model
        """
        values = []
        for u in self.action_space:
            input = np.concatenate((x, [u]))
            values.append(self.model.predict([input]))

        max_index = np.argmax(values).item()
        return self.action_space[max_index]


def J(policy, N, d=None, x=None):
    """
    Compute the expected return of a policy for a state
    """
    if N == 0:
        return 0
    else:
        if d is not None:  # if domain is initialized use it
            u = policy(x)
            new_x, r = d.f(u)
            return r + gamma*J(policy, N-1, d, new_x)
        else:  # else we create it
            d = Domain()
            x = d.initial_state()
            u = policy(x)
            new_x, r = d.f(u)
            return r + gamma*J(policy, N-1, d, new_x)


def build_trajectory(size, from_action_space=False, action_space=None):
    """
    Build a trajctory of four-tuples using ranfom action
    """
    T= []
    d = Domain()
    x = d.initial_state()
    while len(T) < size:
        if not from_action_space:  # continuous action in (-1,1)
            u = d.random_action()
        else:  # choose an action from a custom discrete action space
            u = np.random.choice(action_space, size=1)

        new_x, r = d.f(u)

        # add the four-tuple to the trajectory
        T.append([x, u, r, new_x, d.is_final_state()])

        if d.is_final_state():
            x = d.initial_state()
        else:
            x = new_x

    # shuffle the trajectory
    np.random.shuffle(T)
    return T


def delta(step, model, action_space):
    """
    Compute the temporal difference
    The delta make sense only in discrete action space cases
    """
    if model is None:
        td = step[2]
    else:
        predictions = []
        for u in action_space:  # compute a prediction for each action
            x = np.concatenate((step[3], [u]))
            x = np.array([x])
            if hasattr(model, 'predict'):
                p = model.predict(x).item()
            else:
                p = model(x).item()
            predictions.append(p)

        max_prediction = np.max(predictions)
        x = np.concatenate((step[0], step[1]))
        x = np.array([x])
        if hasattr(model, 'predict'):
            td = step[2] + gamma * max_prediction - model.predict(x).item()
        else:
            td = step[2] + gamma * max_prediction - model(x).item()

    return td


def td_error(model, action_space, nb_approximations=20):
    """
    Computes an estimation of TD-error for a model
    """
    d = Domain()
    s = d.initial_state()
    deltas = []

    for i in range(nb_approximations):
        u = d.random_action()
        next_s, r = d.f(u)

        td = delta([s, u, r, next_s], model, action_space)
        deltas.append(td)

        if d.is_final_state():
            s = d.initial_state()
        else:
            s = next_s

    return np.mean(deltas)
