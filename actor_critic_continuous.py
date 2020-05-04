import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
from domain import Domain


class ActorCriticContinuous:
    """
    Class implementation of TD Advantage Actor-Critic policy search technique
    with continuous action space
    """
    def __init__(self, in_actor, in_critic):
        # actor network
        self.actor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_actor, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, 2),
            nn.ReLU()
        )

        # critic network
        self.critic = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_critic, 10),
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
            s = d.initial_state()

            a_loss = []
            c_loss = []
            rew = 0
            while not d.is_final_state():
                # predict a distribution
                mu, sigma = self.get_distribution(s)

                # sample an action from distribution
                u = torch.randn(1)*sigma + mu

                # clip the value between -1 and 1
                u = u.detach().numpy()
                u = np.clip(u, a_min=-1, a_max=1).item()

                # apply the action and observe next state and reward
                next_s, r = d.f(u)
                rew += r

                # TD target
                V = self.critic(torch.tensor(next_s, dtype=torch.float32))
                V.detach().numpy().item()
                y = r + utils.gamma*V

                # advantage
                j = self.critic(torch.tensor(s, dtype=torch.float32))
                delta = (y - j)**2

                # critic loss
                critic_loss = delta
                print(delta)
                c_loss.append(critic_loss.item())

                # update the critic network
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # actor loss
                log_prob = -((u - mu)**2)/(2*sigma**2) - torch.log(sigma*math.sqrt(2*math.pi))
                actor_loss = -log_prob*delta.detach()
                a_loss.append(actor_loss.item())

                # actor update
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            actor_losses.append(np.mean(a_loss))
            critic_losses.append(np.mean(c_loss))
            rewards.append(rew)

        return actor_losses, critic_losses, rewards


if __name__ == '__main__':
    episode = 1000

    actor_critic = ActorCriticContinuous(in_actor=9, in_critic=9)

    a_losses, c_losses, rewards = actor_critic.train(episode)

    # plt.plot(range(episode), a_losses)
    plt.plot(range(episode), c_losses)
    # plt.plot(range(episode), rewards)
    plt.show()
