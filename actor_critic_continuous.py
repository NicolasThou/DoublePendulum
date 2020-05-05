import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from joblib import dump, load
import utils
from domain import Domain


class ActorCriticContinuous:
    """
    Class implementation of Advantage Actor-Critic policy search technique
    with continuous action space
    """
    def __init__(self):
        # actor network
        self.actor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(9, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 2),
            nn.ReLU()
        )

        # critic network
        self.critic = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(9, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 1)
        )

    def get_distribution(self, s):
        """
        Sample an action using the policy approximation (actor)
        """
        # convert state to tensor
        s = torch.tensor(s, dtype=torch.float32)

        # extract the mean and standard deviation approximated by the network
        mu, sigma = self.actor(s)
        mu, sigma = torch.tanh(mu), F.softplus(sigma) + 1e-05
        return mu, sigma

    def train(self, episode):
        # critic network optimizer
        critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.001)
        critic_optimizer.zero_grad()

        # actor network optimizer
        actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.001)
        actor_optimizer.zero_grad()

        actor_losses = []
        critic_losses = []
        rewards = []

        d = Domain()
        for e in range(episode):
            transitions = []
            log_probs = []
            values = []

            s = d.initial_state()
            while not d.is_final_state():
                # predict the distribution parameters
                mu, sigma = self.get_distribution(s)

                # sample an action from distribution
                u = torch.randn(1)*sigma + mu

                # clip the value between -1 and 1
                u = u.detach().numpy()
                u = np.clip(u, a_min=-1, a_max=1).item()

                # check that u is a number, otherwise go next episode
                if not np.isfinite(u):
                    print('Warning : action not finite number.')
                    break

                # apply the action and observe next state and reward
                next_s, r = d.f(u)
                transitions.append([s, u, r, next_s])

                # value predicted by the critic network
                value = self.critic(torch.tensor(next_s, dtype=torch.float32))
                values.append(value)

                # log used in actor loss
                log_prob = -((u - mu) ** 2) / (2 * sigma ** 2) - torch.log(sigma * math.sqrt(2 * math.pi))
                log_probs.append(log_prob)

                # keep track of next state
                s = next_s

            if not np.isfinite(u):
                continue

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
            critic_loss = (A**2).mean()
            A = A.detach()
            log_probs = torch.stack(log_probs)
            actor_loss = (-log_probs*A).mean()

            # critic update
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # actor update
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # save the loss
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        return actor_losses, critic_losses, rewards


if __name__ == '__main__':
    episode = 1000

    actor_critic = ActorCriticContinuous()

    a_losses, c_losses, rewards = actor_critic.train(episode)
    dump(actor_critic, 'continuous_actor_critic.joblib')
