# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of PPO.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py.

Currently only supports the single-agent case.
"""

import time
from typing import Literal, Optional, Union

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

from iig_rl_benchmark.utils import log_to_csv

INVALID_ACTION_PENALTY = -1e9


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    """A masked categorical."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPOAgent(nn.Module):
    """A PPO agent module."""

    def __init__(self, num_actions, observation_shape, device, layer_init=layer_init, hidden_size=512):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits = self.actor(x)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x),
            probs.probs,
        )
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))

class L2NormLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)
class PPOConditionedOnPolicyRepresentationAgent(nn.Module):
    """A PPO agent module that is conditioned on a policy representation."""

    def __init__(self, num_actions, observation_shape, device, num_policies, policy_embedding_size,layer_init=layer_init, hidden_size=512):
        super().__init__()
        normalization_type: Literal['none', 'layer', 'l2'] = 'layer'
        self.version = f'1_{normalization_type}'
        self.num_policies = num_policies
        if normalization_type == 'none':
            normalization_layer = nn.Identity()
        elif normalization_type == 'layer':
            normalization_layer = nn.LayerNorm(policy_embedding_size, elementwise_affine=True)
        elif normalization_type == 'l2':
            normalization_layer = L2NormLayer(dim=1, eps=1e-12)
        else:
            raise ValueError(f"Invalid normalization type: {normalization_type}")
        self.embedding_prenorm = nn.Embedding(num_policies, policy_embedding_size)
        self.embedding_norm = normalization_layer
        self.policy_representation_embedding = nn.Sequential(
            self.embedding_prenorm,
            self.embedding_norm,
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod() + policy_embedding_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod() + policy_embedding_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x, policy_index):
        # Expand policy_index to match batch size
        batch_size = x.shape[0]
        policy_index_expanded = policy_index.expand(batch_size)
        return self.critic(torch.cat([x, self.policy_representation_embedding(policy_index_expanded)], dim=1))

    def get_action_and_value(self, x, policy_index=None, embedding=None,legal_actions_mask=None, action=None):
        assert (policy_index is None) != (embedding is None), "Exactly one of policy_index or embedding must be provided"
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        # Expand policy_index to match batch size
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        if policy_index is not None:
            policy_index_expanded = policy_index.expand(batch_size)
            nn_input = torch.cat([x, self.policy_representation_embedding(policy_index_expanded)], dim=1)
        else:
            embedding_expanded = embedding.expand(batch_size, -1)
            nn_input = torch.cat([x, embedding_expanded], dim=1)

        logits = self.actor(nn_input)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(nn_input),
            probs.probs,
        )
    
    def save(self, path):
        torch.save({'version': self.version, 'actor': self.actor.state_dict(), 'policy_representation_embedding': self.policy_representation_embedding.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        version = checkpoint.get('version', None)
        if version is None or version != self.version:
            raise ValueError(f"Version mismatch: {version} != {self.version}")
        self.actor.load_state_dict(checkpoint['actor'])
        self.policy_representation_embedding.load_state_dict(checkpoint['policy_representation_embedding'])


class PPOAtariAgent(nn.Module):
    """A PPO Atari agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super(PPOAtariAgent, self).__init__()
        # Note: this network is intended for atari games, taken from
        # https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.num_actions = num_actions
        self.device = device
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = CategoricalMasked(
            logits=logits, masks=legal_actions_mask, mask_value=self.mask_value
        )

        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            probs.probs,
        )


def legal_actions_to_mask(legal_actions_list, num_actions):
    """Converts a list of legal actions to a mask.

    The mask has size num actions with a 1 in a legal positions.

    Args:
      legal_actions_list: the list of legal actions
      num_actions: number of actions (width of mask)

    Returns:
      legal actions mask.
    """
    legal_actions_mask = torch.zeros(
        (len(legal_actions_list), num_actions), dtype=torch.bool
    )
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask


class PPO(nn.Module):
    """PPO Agent implementation in PyTorch.

    See open_spiel/python/examples/ppo_example.py for an usage example.

    Note that PPO runs multiple environments concurrently on each step (see
    open_spiel/python/vector_env.py). In practice, this tends to improve PPO's
    performance. The number of parallel environments is controlled by the
    num_envs argument.
    """

    def __init__(
        self,
        input_shape,
        num_actions,
        num_players,
        num_envs=1,
        steps_per_batch=128,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_size=512,
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        device="cpu",
        agent_fn: Union[PPOAgent, PPOAtariAgent, PPOConditionedOnPolicyRepresentationAgent]=PPOAtariAgent,
        neupl_ppo_kwargs: Optional[dict]=None,
        neupl_ppo_policy_index: Optional[int]=None,
        network: Optional[nn.Module]=None,
        log_file=None,
        **kwargs,
    ):
        super().__init__()
        assert (agent_fn == PPOConditionedOnPolicyRepresentationAgent) == (neupl_ppo_policy_index is not None) == (neupl_ppo_kwargs is not None), "neupl_ppo_policy_index and neupl_ppo_kwargs must be provided if agent_fn is PPOConditionedOnPolicyRepresentationAgent"

        self.input_shape = (np.array(input_shape).prod(),)
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.log_file = log_file

        # Training settings
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.batch_size = self.num_envs * self.steps_per_batch
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.anneal_lr = kwargs.get("anneal_lr", False)

        # Loss function
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Initialize networks
        if neupl_ppo_policy_index is not None:
            self.neupl_ppo_policy_index = torch.tensor(neupl_ppo_policy_index).to(device).unsqueeze(0)
        if network is None:
            self.network = agent_fn(self.num_actions, self.input_shape, device, hidden_size=hidden_size, **(neupl_ppo_kwargs or {})).to(device)
        else:
            self.network = network
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

        # Initialize training buffers
        self.legal_actions_mask = torch.zeros(
            (self.steps_per_batch, self.num_envs, self.num_actions), dtype=torch.bool
        ).to(device)
        self.obs = torch.zeros(
            (self.steps_per_batch, self.num_envs, *self.input_shape)
        ).to(device)
        self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.current_players = torch.zeros((self.steps_per_batch, self.num_envs)).to(
            device
        )

        # Initialize counters
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()

    def get_value(self, x):
        if isinstance(self.network, PPOConditionedOnPolicyRepresentationAgent):
            return self.network.get_value(x, self.neupl_ppo_policy_index)
        else:
            return self.network.get_value(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        if isinstance(self.network, PPOConditionedOnPolicyRepresentationAgent):
            return self.network.get_action_and_value(x, policy_index=self.neupl_ppo_policy_index, legal_actions_mask=legal_actions_mask, action=action)
        else:
            return self.network.get_action_and_value(x, legal_actions_mask, action)

    def step(self, time_step, is_evaluation=False):
        if is_evaluation:
            with torch.no_grad():
                legal_actions_mask = legal_actions_to_mask(
                    [
                        ts.observations["legal_actions"][ts.current_player()]
                        for ts in time_step
                    ],
                    self.num_actions,
                ).to(self.device)
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                ts.observations["info_state"][ts.current_player()],
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                action, _, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )
                return [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]
        else:
            with torch.no_grad():
                # act
                obs = torch.Tensor(
                    np.array(
                        [
                            np.reshape(
                                ts.observations["info_state"][ts.current_player()],
                                self.input_shape,
                            )
                            for ts in time_step
                        ]
                    )
                ).to(self.device)
                legal_actions_mask = legal_actions_to_mask(
                    [
                        ts.observations["legal_actions"][ts.current_player()]
                        for ts in time_step
                    ],
                    self.num_actions,
                ).to(self.device)
                current_players = torch.Tensor(
                    [ts.current_player() for ts in time_step]
                ).to(self.device)

                action, logprob, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )

                # store
                self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
                self.obs[self.cur_batch_idx] = obs
                self.actions[self.cur_batch_idx] = action
                self.logprobs[self.cur_batch_idx] = logprob
                self.values[self.cur_batch_idx] = value.flatten()
                self.current_players[self.cur_batch_idx] = current_players

                agent_output = [
                    StepOutput(action=a.item(), probs=p)
                    for (a, p) in zip(action, probs)
                ]
                return agent_output

    def post_step(self, reward, done):
        self.rewards[self.cur_batch_idx] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)

        self.total_steps_done += self.num_envs
        self.cur_batch_idx += 1

    def learn(self, time_step):
        next_obs = torch.Tensor(
            np.array(
                [
                    np.reshape(
                        ts.observations["info_state"][ts.current_player()],
                        self.input_shape,
                    )
                    for ts in time_step
                ]
            )
        ).to(self.device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    nextvalues = (
                        next_value
                        if t == self.steps_per_batch - 1
                        else self.values[t + 1]
                    )
                    nextnonterminal = 1.0 - self.dones[t]
                    delta = (
                        self.rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = (
                        next_value if t == self.steps_per_batch - 1 else returns[t + 1]
                    )
                    nextnonterminal = 1.0 - self.dones[t]
                    returns[t] = (
                        self.rewards[t] + self.gamma * nextnonterminal * next_return
                    )
                advantages = returns - self.values

        # flatten the batch
        b_legal_actions_mask = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_playersigns = -2.0 * self.current_players.reshape(-1) + 1.0
        b_advantages *= b_playersigns

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
                    b_obs[mb_inds],
                    legal_actions_mask=b_legal_actions_mask[mb_inds],
                    action=b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.entropy_coef * entropy_loss
                    + v_loss * self.value_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Commented this out because it takes too much disk space for the large sweep
        # log_data = {
        #     "steps": self.total_steps_done,
        #     "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
        #     "losses/value_loss": v_loss.item(),
        #     "losses/policy_loss": pg_loss.item(),
        #     "losses/entropy": entropy_loss.item(),
        #     "losses/old_approx_kl": old_approx_kl.item(),
        #     "losses/approx_kl": approx_kl.item(),
        #     "losses/clipfrac": np.mean(clipfracs),
        #     "losses/explained_variance": explained_var,
        #     "charts/SPS": int(
        #         self.total_steps_done / (time.time() - self.start_time)
        #     ),
        # }
        # log_to_csv(log_data, self.log_file)

        # Update counters
        self.updates_done += 1
        self.cur_batch_idx = 0

    def save(self, path):
        """Saves the actor weights to path"""
        self.network.save(path)

    def load(self, path):
        """Loads weights from actor checkpoint"""
        self.network.load(path)

    def anneal_learning_rate(self, update, num_total_updates):
        # Annealing the rate
        frac = max(0, 1.0 - (update / num_total_updates))
        if frac < 0:
            raise ValueError("Annealing learning rate to < 0")
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow
