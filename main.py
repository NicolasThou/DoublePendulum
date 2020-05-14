import time
import warnings
from domain import Domain
import utils
from actor_critic_discrete import *
from actor_critic_continuous import *
from fitted_q_iteration import *


# disable warnings
warnings.filterwarnings("ignore")


# create a discrete actor critic and train it for 10,000 episodes
discrete_ac = ActorCriticDiscrete(action_space=[-1, 1])
discrete_ac.train(10000)


# create a discrete actor critic and train it
continuous_ac = ActorCriticContinuous()
continuous_ac.train(10000)


# create an Extra-Tree and train it with 2-dimensional action space
tree = train_ExtraTree(action_space=[-1, 1])


# compute the expected reward of the discrete actor-critic model
j_list = []
for i in range(1000):
    j_list.append(utils.J(discrete_ac, 100))
print(f'Expected return of discrete actor critic : {np.mean(j_list)}')


# run a simulation using the continuous actor-critic as policy
d = Domain()
d.env.render()
s = d.initial_state()
while not d.is_final_state():  # we continue until we reach a final state
    u = continuous_ac(s)
    next_s, r = d.f(u)
    time.sleep(0.01)  # let time for the rendering
    if d.is_final_state():
        s = d.initial_state()
    else:
        s = next_s


# compute the expected reward of the Extra-Tree model
# be careful : the action space used here has to match the action space used for the training !
mu = Policy(tree, action_space=[-1, 1])
j_list = []
for i in range(1000):  # compute 1000 episode to have the expected return as an average
    j_list.append(utils.J(mu, 100))
print(f'Expected return of Extra-Tree : {np.mean(j_list)}')
