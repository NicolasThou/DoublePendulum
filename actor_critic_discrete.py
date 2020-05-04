import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
from domain import Domain


action_space = [-1, -0.5, -0.25, 0.25, 0.5, 1]


class ActorCriticDiscrete:
    """
    Class implementation of Actor-Critic policy search technique
    with discrete action space
    """
    def __init__(self, in_actor, out_actor, in_critic):
        # actor network
        self.actor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_actor, 10),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(10, out_actor),
            nn.Softmax()
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

        # extract distribution over action space produced by the actor network
        y = self.actor(s)

        return y

    def train(self, episode=10):
        # optimizer of critic network
        critic_optimizer = optim.SGD(self.critic.parameters(), lr=0.01)
        critic_optimizer.zero_grad()

        # actor optimizer
        actor_optimizer = optim.SGD(self.actor.parameters(), lr=0.01)
        actor_optimizer.zero_grad()

        actor_losses = []
        critic_losses = []
        rewards = []
        d = Domain()
        for e in range(episode):
            transitions = []
            log_probs = []
            advantages = []
            rew = 0

            s = d.initial_state()
            while not d.is_final_state():
                p = self.get_distribution(s)

                # get action with highest probability
                idx = torch.argmax(p).detach().numpy().item()
                u = action_space[idx]

                # save probability for -log
                log_probs.append(torch.log(p[idx]))

                # apply the action and observe next state and reward
                next_s, r = d.f(u)

                transitions.append([s, u, r, next_s])
                s = next_s
                rew += r

                V = self.critic(torch.tensor(next_s, dtype=torch.float32))
                V = V.detach().numpy().item()
                y = r + utils.gamma*V

                advantages.append((y - self.critic(torch.tensor(s, dtype=torch.float32)))**2)
                # print(advantages[-1])

                # update the critic network using TD error
                critic_optimizer.zero_grad()
                critic_loss = (y - self.critic(torch.tensor(s, dtype=torch.float32)))**2
                critic_losses.append(critic_loss.item())
                critic_loss.backward()
                critic_optimizer.step()

            rewards.append(rew)

            # actor loss and update
            actor_optimizer.zero_grad()
            actor_loss = (-sum(log_probs)).mean()
            actor_losses.append((actor_loss.item()))
            actor_loss.backward()
            actor_optimizer.step()

        return actor_losses, critic_losses, rewards


if __name__ == '__main__':
    episode = 1000

    actor_critic = ActorCriticDiscrete(in_actor=9, out_actor=len(action_space), in_critic=9)

    a_losses, c_losses, r = actor_critic.train(episode)

    plt.plot(range(len(a_losses)), a_losses)
    # plt.plot(range(len(c_losses)), c_losses)
    # plt.plot(range(len(r), r)
    plt.show()
