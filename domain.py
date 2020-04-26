import gym
import pybulletgym


class Domain:
    """
    Class that implements the dynamic of the doman
    """
    def __init__(self):
        self.env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
        self.done = None

    def __del__(self):
        self.env.close()

    def initial_state(self):
        """
        Return an initial state
        """
        state = self.env.reset()
        self.done = False
        return state

    def f(self, action):
        """
        Dynamic of the domain
            The state isn't taking in consideration since
            the current state is saved in the library
        """
        state, reward, self.done, _ = self.env.step(action)
        return state, reward

    def is_final_state(self):
        """
        Return true if a final state is reached
        """
        return self.done

    def random_action(self, state=None):
        """
        Return a random action
        """
        return self.env.action_space.sample()


if __name__ == '__main__':
    d = Domain()
    s = d.initial_state()
    print(f'initial state: {s}')
    for t in range(1000):
        d.env.render()
        u = d.random_action()
        s, r = d.f(u)
        print(r)
        if d.is_final_state():
            print('final state reached')
            break
