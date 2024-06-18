import gym
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from torch.distributions.categorical import Categorical
from torch.optim import Adam


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
  """An MLP that represents a trainable policy."""
  # Build a feedforward neural network.
  layers = []
  for j in range(len(sizes) - 1):
    act = activation if j < len(sizes) - 2 else output_activation
    input_dim, output_dim = sizes[j], sizes[j + 1]
    layers += [nn.Linear(input_dim, output_dim), act()]
  return nn.Sequential(*layers)


def train(*,
          env_name='CartPole-v0',
          hidden_sizes=[32],
          lr=1e-2,
          epochs=50,
          batch_size=5000,
          render=False):
  # make environment, check spaces, get obs / act dims
  env = gym.make(env_name)
  assert isinstance(env.observation_space, spaces.Box), \
    "This example only works for envs with continuous state spaces."
  assert isinstance(env.action_space, spaces.Discrete), \
    "This example only works for envs with discrete action spaces."

  obs_dim = env.observation_space.shape[0]
  n_acts = env.action_space.n

  # make core of policy network
  logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

  # make function to compute action distribution
  def get_policy(obs):
    """Return distribution of actions given states/observations."""
    logits = logits_net(obs)
    return Categorical(logits=logits)

  # make action selection function (outputs int actions, sampled from policy)
  def get_action(obs):
    """Sample from the action distribution."""
    return get_policy(obs).sample().item()

  # make loss function whose gradient, for the right data, is policy gradient
  def compute_loss(obs, act, weights):
    """Compute weighted negative log prob for an action.

    Args:
        obs: Observations/states from time 0, 1, ..., T.
        act: Actions at time 0, 1, ..., T.
        weights: In naive policy gradient, weight is the return of the
         trajectory.
    Returns:
        The loss of a trajectory.
    """
    log_prob = get_policy(obs).log_prob(act)
    return -(log_prob * weights).mean()

  # make optimizer
  optimizer = Adam(logits_net.parameters(), lr=lr)

  # for training policy
  def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []  # for observations
    batch_acts = []  # for actions
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rets = []  # for measuring episode returns
    batch_lens = []  # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over
    ep_rews = []  # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

      # rendering
      if (not finished_rendering_this_epoch) and render:
        env.render()

      # save obs
      batch_obs.append(obs.copy())

      # act in the environment
      act = get_action(torch.as_tensor(obs, dtype=torch.float32))
      obs, rew, done, _ = env.step(act)

      # save action, reward
      batch_acts.append(act)
      ep_rews.append(rew)  # immediate reward

      if done:
        # if episode is over, record info about episode
        episode_return, episode_length = sum(ep_rews), len(ep_rews)
        batch_rets.append(episode_return)
        batch_lens.append(episode_length)

        # the weight for each logprob(a|s) is R(tau)
        batch_weights += [episode_return] * episode_length

        # reset episode-specific variables
        obs, done, ep_rews = env.reset(), False, []

        # won't render again this epoch
        finished_rendering_this_epoch = True

        # end experience loop if we have enough of it
        if len(batch_obs) > batch_size:
          break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(
      obs=torch.as_tensor(batch_obs, dtype=torch.float32),
      act=torch.as_tensor(batch_acts, dtype=torch.int32),
      weights=torch.as_tensor(batch_weights, dtype=torch.float32)
      )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens

  # training loop
  for i in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t episode_length: %.3f' %
          (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--lr', type=float, default=1e-2)
  args = parser.parse_args()
  print('\nUsing simplest formulation of policy gradient.\n')
  train(env_name=args.env_name, render=args.render, lr=args.lr, epochs=2)
