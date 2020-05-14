import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
import time
import utils
from domain import Domain


class ActorCriticDiscrete:
    """
    Class implementation of Actor-Critic policy search technique
    with discrete action space
    """

    def __init__(self, action_space):
        # action_space is the discretization of the continuous action space
        # it must be given as a list of possible actions
        self.action_space = action_space

        # actor network
        self.actor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(9, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, len(self.action_space)),
            nn.Softmax()
        )

        # critic network
        self.critic = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(9, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 1)
        )

    def __call__(self, x):
        """
        Make a prediction for a state
        """
        x = torch.tensor(x, dtype=torch.float32)

        # extract distribution on action space
        p = self.actor(x)

        # sometimes predictions returned are 'nan'
        # in those case we return a random action
        if not np.isfinite(p.detach().numpy()).all():
            return np.random.choice(self.action_space, size=1).item()
        else:
            # return action with maximum probability
            idx = torch.argmax(p).detach().numpy().item()
            return self.action_space[idx]

    def get_distribution(self, s):
        """
        Sample an action using the policy approximation (actor)
        """
        s = torch.tensor(s, dtype=torch.float32)

        # extract distribution over action space produced by the actor network
        y = self.actor(s)

        return y

    def train(self, episode=10):
        # optimizer of critic network
        critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.001)
        critic_optimizer.zero_grad()

        # actor optimizer
        actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.001)
        actor_optimizer.zero_grad()

        actor_losses = []
        critic_losses = []
        rewards = []

        d = Domain()
        for e in range(episode):
            print(f'========== episode {e} ==========')
            transitions = []
            log_probs = []
            values = []

            s = d.initial_state()
            while not d.is_final_state(): # episode terminates when we reach a final state
                p = self.get_distribution(s)

                if not np.isfinite(p.detach().numpy()).all():
                    print('Warning : probabilities not finite numbers.')
                    break

                # get action with highest probability
                idx = torch.argmax(p).detach().numpy().item()
                u = self.action_space[idx]

                # apply the action and observe next state and reward
                next_s, r = d.f(u)
                transitions.append([s, u, r, next_s])

                # save log probability
                log_probs.append(torch.log(p[idx]))

                value = self.critic(torch.tensor(s, dtype=torch.float32))
                values.append(value)

                # keep track of next state
                s = next_s

            if not np.isfinite(p.detach().numpy()).all():
                continue

            # save the sum of episode rewards
            episode_rewards = np.array(transitions)
            episode_rewards = episode_rewards[:, 2].tolist()
            rewards.append(sum(episode_rewards))

            R = 0
            A = torch.zeros(len(values))
            for t in reversed(range(len(transitions))):
                R = transitions[t][2] + utils.gamma * R
                A[t] = R

            # advantage
            A = A - torch.cat(values)

            # actor and critic loss
            critic_loss = (A ** 2).mean()
            A = A.detach()
            log_probs = torch.stack(log_probs)
            actor_loss = (-log_probs * A).mean()

            # update the critic network
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # update tha actor network
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # save the loss
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            print(f' critic loss : {critic_losses[-1]} | actor loss : {actor_losses[-1]}')

        return actor_losses, critic_losses, rewards


if __name__ == '__main__':
    model = load('models/discrete_actor_critic 2-dimensional.joblib')
    d = Domain()
    d.env.render()
    s = d.initial_state()
    while True:
        u = model(s)
        next_s, r = d.f(u)
        time.sleep(0.01)
        if d.is_final_state():
            s = d.initial_state()
        else:
            s = next_s
