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

"""An implementation of PPG.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg.py and
https://github.com/vwxyzjn/ppg-implementation-details/blob/main/ppg_atari.py.

Currently only suppgrts the single-agent case.
"""

import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

from open_spiel.python.rl_agent import StepOutput

from utils import log_to_csv

INVALID_ACTION_PENALTY = -1e9


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = torch.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert torch.equal(a, c)


class CategoricalMasked(Categorical):
    """A masked categorical."""

    # pylint: disable=dangerous-default-value
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None
    ):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class PPGAgent(nn.Module):
    """A PPG agent module."""

    def __init__(self, num_actions, observation_shape, device):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.policy_base = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.aux_critic_head = layer_init(nn.Linear(512, 1), std=1.0)
        self.actor = nn.Sequential(
            self.policy_base,
            self.actor_head,
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

    # PPG logic
    def get_pi_value_and_aux_value(self, x, legal_actions_mask):
        hidden = self.policy_base(x)
        return (
            CategoricalMasked(
                logits=self.actor_head(hidden),
                masks=legal_actions_mask,
                mask_value=self.mask_value,
            ),
            self.critic(x),
            self.aux_critic_head(hidden),
        )

    def get_pi(self, x, legal_actions_mask):
        hidden = self.policy_base(x)
        return CategoricalMasked(
            logits=self.actor_head(hidden),
            masks=legal_actions_mask,
            mask_value=self.mask_value,
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


class PPG(nn.Module):
    """PPG Agent implementation in PyTorch.

    See open_spiel/python/examples/ppg_example.py for an usage example.

    Note that PPG runs multiple environments concurrently on each step (see
    open_spiel/python/vector_env.py). In practice, this tends to improve PPG's
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
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        n_iteration=32,
        e_policy=1,
        v_value=1,
        e_auxiliary=6,
        beta_clone=1.0,
        num_aux_rollouts=4,
        n_aux_grad_accum=1,
        device="cpu",
        agent_fn=PPGAgent,
        log_file=None,
    ):
        super().__init__()

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

        # PPG params
        self.n_iteration = n_iteration
        self.e_policy = e_policy
        self.v_value = v_value
        self.e_auxiliary = e_auxiliary
        self.beta_clone = beta_clone
        self.num_aux_rollouts = num_aux_rollouts
        self.n_aux_grad_accum = n_aux_grad_accum

        # PPG
        self.aux_batch_rollouts = int(self.num_envs * self.n_iteration)

        # Initialize networks
        self.network = agent_fn(self.num_actions, self.input_shape, device).to(device)
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
        # PPG
        self.aux_obs = torch.zeros(
            (self.steps_per_batch, self.aux_batch_rollouts, *self.input_shape)
        ).to(device)  # , dtype=torch.uint8)  # Saves lot system RAM
        self.aux_legal_actions_mask = torch.zeros(
            (self.steps_per_batch, self.aux_batch_rollouts, self.num_actions),
            dtype=torch.bool,
        ).to(device)
        self.aux_returns = torch.zeros(
            (self.steps_per_batch, self.aux_batch_rollouts)
        ).to(device)

        # Initialize counters
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()

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
                action, _, _, value, probs = self.network.get_action_and_value(
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

                action, logprob, _, value, probs = self.network.get_action_and_value(
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

    def learn(self, time_step, update):
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
            next_value = self.network.get_value(next_obs).reshape(1, -1)
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

                _, newlogprob, entropy, newvalue, _ = self.network.get_action_and_value(
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

        # PPG Storage - Rollouts are saved without flattening for sampling full rollouts later:
        # "update" ranges from 0 to self.n_iteration - 1
        storage_slice = slice(self.num_envs * update, self.num_envs * (update + 1))
        self.aux_obs[:, storage_slice] = self.obs.cpu().clone()  # .to(torch.uint8)
        self.aux_legal_actions_mask[:, storage_slice] = self.legal_actions_mask.clone()
        self.aux_returns[:, storage_slice] = returns.cpu().clone()

    def learn_auxiliary(self):
        aux_inds = np.arange(self.aux_batch_rollouts)

        # Build the old policy on the aux buffer before distilling to the network
        aux_pi = torch.zeros(
            (self.steps_per_batch, self.aux_batch_rollouts, self.num_actions)
        )
        for i, start in enumerate(
            range(0, self.aux_batch_rollouts, self.num_aux_rollouts)
        ):
            end = start + self.num_aux_rollouts
            aux_minibatch_ind = aux_inds[start:end]
            m_aux_obs = (
                self.aux_obs[:, aux_minibatch_ind].to(torch.float32).to(self.device)
            )
            m_obs_shape = m_aux_obs.shape
            m_aux_obs = flatten01(m_aux_obs)
            m_aux_lam = self.aux_legal_actions_mask[:, aux_minibatch_ind].to(
                self.device
            )
            m_aux_lam = flatten01(m_aux_lam)
            with torch.no_grad():
                pi_logits = (
                    self.network.get_pi(m_aux_obs, m_aux_lam).logits.cpu().clone()
                )
            aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2])
            del m_aux_obs

        for auxiliary_update in range(1, self.e_auxiliary + 1):
            np.random.shuffle(aux_inds)
            for i, start in enumerate(
                range(0, self.aux_batch_rollouts, self.num_aux_rollouts)
            ):
                end = start + self.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]
                try:
                    m_aux_obs = self.aux_obs[:, aux_minibatch_ind].to(self.device)
                    m_obs_shape = m_aux_obs.shape
                    m_aux_obs = flatten01(
                        m_aux_obs
                    )  # Sample full rollouts for PPG instead of random indexes
                    m_aux_lam = self.aux_legal_actions_mask[:, aux_minibatch_ind].to(
                        self.device
                    )
                    m_aux_lam = flatten01(m_aux_lam)
                    m_aux_returns = (
                        self.aux_returns[:, aux_minibatch_ind]
                        .to(torch.float32)
                        .to(self.device)
                    )
                    m_aux_returns = flatten01(m_aux_returns)

                    new_pi, new_values, new_aux_values = (
                        self.network.get_pi_value_and_aux_value(m_aux_obs, m_aux_lam)
                    )

                    new_values = new_values.view(-1)
                    new_aux_values = new_aux_values.view(-1)
                    old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(
                        self.device
                    )
                    old_pi = Categorical(logits=old_pi_logits)
                    kl_loss = torch.distributions.kl_divergence(old_pi, new_pi).mean()

                    real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                    aux_value_loss = (
                        0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                    )
                    joint_loss = aux_value_loss + self.beta_clone * kl_loss

                    loss = (joint_loss + real_value_loss) / self.n_aux_grad_accum
                    loss.backward()

                    if (i + 1) % self.n_aux_grad_accum == 0:
                        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()  # This cannot be outside, else gradients won't accumulate

                except RuntimeError as e:
                    raise Exception(
                        "if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory"
                    ) from e

                del m_aux_obs, m_aux_returns

        # Commented this out because it takes too much disk space for the large sweep
        # log_data = {
        #     "steps": self.total_steps_done,
        #     "losses/aux/kl_loss": kl_loss.mean().item(),
        #     "losses/aux/aux_value_loss": aux_value_loss.item(),
        #     "losses/aux/real_value_loss": real_value_loss.item(),
        # }
        # log_to_csv(log_data, self.log_file[:-4] + '_aux.csv')

    def save(self, path):
        """Saves the actor weights to path"""
        torch.save(self.network.actor.state_dict(), path)

    def load(self, path):
        """Loads weights from actor checkpoint"""
        self.network.actor.load_state_dict(torch.load(path))

    def anneal_learning_rate(self, update, num_total_updates):
        # Annealing the rate
        frac = max(0, 1.0 - (update / num_total_updates))
        if frac < 0:
            raise ValueError("Annealing learning rate to < 0")
        lrnow = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow
