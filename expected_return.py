import numpy as np
from joblib import load
import warnings
import utils


class ActorCriticPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(x)


class ExtraTreePolicy:
    """
    Action space is discrete
    """
    def __init__(self, model, action_space):
        self.model = model
        self.action_space = action_space

    def __call__(self, x):
        values = []
        for u in self.action_space:
            input = np.concatenate((x, [u]))
            values.append(self.model.predict([input]))

        max_index = np.argmax(values).item()
        return self.action_space[max_index]


if __name__ == '__main__':
    # disable warning
    warnings.filterwarnings("ignore")

    model = load('models/discrete_actor_critic 6-dimensional.joblib')
    mu = ActorCriticPolicy(model)

    j_list = []
    for i in range(1000):
        j_list.append(utils.J(mu, 100))

    print(np.mean(j_list))
