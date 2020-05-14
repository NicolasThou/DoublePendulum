import domain
import utils
import expected_return
import actor_critic_discrete
import actor_critic_continuous
import fitted_q_iteration


# create a discrete actor critic and train it
discrete_ac = actor_critic_discrete.ActorCriticDiscrete(action_space=[-1, 1])
discrete_ac.train(10000)

# create a discrete actor critic and train it
continuous_ac = actor_critic_continuous.ActorCriticContinuous()
continuous_ac.train(10000)

# create an Extra-Tree and train it
tree = fitted_q_iteration.train_ExtraTree(action_space=[-1, 1])
