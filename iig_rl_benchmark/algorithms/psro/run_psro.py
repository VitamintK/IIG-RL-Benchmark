# Copyright 2019 DeepMind Technologies Limited
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

"""Example running PSRO on OpenSpiel Sequential games.

To reproduce results from (Muller et al., "A Generalized Training Approach for
Multiagent Learning", ICLR 2020; https://arxiv.org/abs/1909.12823), run this
script with:
  - `game_name` in ['kuhn_poker', 'leduc_poker']
  - `num_players` in [2, 3, 4, 5]
  - `meta_strategy_method` in ['alpharank', 'uniform', 'nash', 'prd']
  - `rectifier` in ['', 'rectified']

The other parameters keeping their default values.
"""

import time

import numpy as np

# pylint: disable=g-bad-import-order
import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from algorithms.psro import psro
from algorithms.psro import rl_oracle
from open_spiel.python import rl_agent
from algorithms.ppo.ppo import PPOAgent

from algorithms.psro import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
import argparse
import torch
from torch.distributions import Categorical
import os


class PSRORLAgent(rl_agent.AbstractAgent):
    def __init__(
        self, agent_type, models, strategy_profile, player_id, n_obs, n_actions
    ):
        self.agent_type = agent_type
        self.player_id = player_id
        self.models = models
        self.strategy_profile = strategy_profile
        self.num_strategies = len(self.models)
        self.num_actions = n_actions
        self.num_obs = n_obs
        self.cur_model = self.models[
            np.random.choice(self.num_strategies, p=self.strategy_profile)
        ]

    def step(self, time_step, is_evaluation=False):
        if time_step.first():
            self.cur_model = self.models[
                np.random.choice(self.num_strategies, p=self.strategy_profile)
            ]
        obs = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        legal_actions = np.array(legal_actions)
        legal_actions_mask = torch.zeros((self.num_actions,), dtype=torch.bool)
        legal_actions_mask[legal_actions] = True
        logits = self.cur_model(torch.tensor(np.array(obs), dtype=torch.float))
        if self.agent_type == "dqn":
            logits = torch.where(legal_actions_mask.bool(), logits, -1e6)
            action = torch.argmax(logits)
            probs = np.zeros(self.num_actions)
            probs[action] = 1.0
            return rl_agent.StepOutput(action=int(action), probs=probs)
        elif self.agent_type == "ppo":
            action_probs = torch.softmax(logits, dim=1)
            probs = np.zeros(self.n_actions)
            probs[legal_actions] = action_probs[0][legal_actions].detach().numpy()
            probs /= sum(probs)
            action = np.random.choice(len(probs), p=probs)
            return rl_agent.StepOutput(action=action, probs=probs)

    def _create_model_action(self, info_state, cur_model):
        action_values = cur_model(info_state)
        return action_values

    def get_model(self):
        # we need to do this funky lambda expression because we need to pass a list of models without indexing the models
        # as we go. So we create a lambda function that will index the models as we go
        models = [
            lambda info_state, cur_model=cur_model: self._create_model_action(
                info_state, cur_model
            )
            for cur_model in self.models
        ]
        return {"models": models, "weights": self.strategy_profile}


class RunPSRO:
    def __init__(self, args, game, expl_callback=None):
        self.args = args.algorithm
        self.meta_args = args
        self.game = game
        self.expl_callback = expl_callback

    def run(self):
        # game = pyspiel.load_game_as_turn_based(self.args.game)
        env = rl_environment.Environment(self.game)
        self.num_actions = env.action_spec()["num_actions"]
        self.info_state_size = env.observation_spec()["info_state"][0]

        # Initialize oracle and agents
        # with tf.Session() as sess:
        if self.args.oracle_type == "dqn":
            oracle, agents = self.init_dqn_responder(env)
        elif self.args.oracle_type == "PG":
            oracle, agents = self.init_pg_responder(env)
        elif self.args.oracle_type == "BR":
            oracle, agents = self.init_br_responder(env)
        elif self.args.oracle_type == "ppo":
            oracle, agents = self.init_ppo_responder(env)
        else:
            raise ValueError(f"Oracle type {self.args.oracle_type} not recognized.")

        self.gpsro_looper(env, oracle, agents, args=self.args)

    def init_ppo_responder(self, env):
        """Initializes the Policy Gradient-based responder and agents."""
        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        agent_class = rl_policy.PPOPolicy

        agent_kwargs = {
            "info_state_size": info_state_size,
            "num_actions": num_actions,
            "steps_per_batch": self.args.inner_rl_ppo_steps_per_batch,
            "num_minibatches": self.args.inner_rl_ppo_num_minibatches,
            "update_epochs": self.args.inner_rl_ppo_update_epochs,
            "learning_rate": self.args.inner_rl_ppo_learning_rate,
            "gae": self.args.inner_rl_ppo_gae,
            "gamma": self.args.inner_rl_ppo_gamma,
            "gae_lambda": self.args.inner_rl_ppo_gae_lambda,
            "normalize_advantages": self.args.inner_rl_ppo_normalize_advantages,
            "clip_coef": self.args.inner_rl_ppo_clip_coef,
            "clip_vloss": self.args.inner_rl_ppo_clip_vloss,
            "entropy_coef": self.args.inner_rl_ppo_entropy_coef,
            "value_coef": self.args.inner_rl_ppo_value_coef,
            "max_grad_norm": self.args.inner_rl_ppo_max_grad_norm,
            "target_kl": self.args.inner_rl_ppo_target_kl,
            "anneal_lr": self.args.inner_rl_ppo_anneal_lr,
            "use_wandb": False,
            "agent_fn": PPOAgent,
            "oracle_type": "ppo",
        }
        oracle = rl_oracle.RLOracle(
            env,
            agent_class,
            agent_kwargs,
            number_training_episodes=self.args.number_training_episodes,
            self_play_proportion=self.args.self_play_proportion,
            sigma=self.args.sigma,
        )

        agents = [
            agent_class(  # pylint: disable=g-complex-comprehension
                env, player_id, **agent_kwargs
            )
            for player_id in range(2)
        ]
        for agent in agents:
            agent.freeze()
        return oracle, agents

    def init_br_responder(self, env):
        """Initializes the tabular best-response based responder and agents."""
        random_policy = policy.TabularPolicy(env.game)
        oracle = best_response_oracle.BestResponseOracle(
            game=env.game, policy=random_policy
        )
        agents = [random_policy.__copy__() for _ in range(self.args.num_players)]
        return oracle, agents

    def init_dqn_responder(self, env):
        """Initializes the Policy Gradient-based responder and agents."""
        state_representation_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        agent_class = rl_policy.DQNPolicy
        agent_kwargs = {
            "state_representation_size": state_representation_size,
            "num_actions": num_actions,
            "hidden_layers_sizes": [self.args.hidden_layer_size]
            * self.args.n_hidden_layers,
            "batch_size": self.args.inner_rl_agent.batch_size,
            "learning_rate": self.args.inner_rl_agent.learning_rate,
            "update_target_network_every": self.args.inner_rl_agent.update_target_network_every,
            "learn_every": self.args.inner_rl_agent.learn_every,
            "optimizer_str": self.args.inner_rl_agent.optimizer_str,
        }
        oracle = rl_oracle.RLOracle(
            env,
            agent_class,
            agent_kwargs,
            number_training_episodes=self.args.number_training_episodes,
            self_play_proportion=self.args.self_play_proportion,
            sigma=self.args.sigma,
            oracle_type=self.args.oracle_type,
        )

        agents = [
            agent_class(  # pylint: disable=g-complex-comprehension
                env, player_id, **agent_kwargs
            )
            for player_id in range(2)  # self.args.num_players)
        ]
        for agent in agents:
            agent.freeze()
        return oracle, agents

    def gpsro_looper(self, env, oracle, agents, args):
        """Initializes and executes the GPSRO training loop."""
        sample_from_marginals = True  # TODO(somidshafiei) set False for alpharank
        training_strategy_selector = (
            args.training_strategy_selector or strategy_selectors.probabilistic
        )

        g_psro_solver = psro.PSROSolver(
            env.game,
            oracle,
            initial_policies=agents,
            training_strategy_selector=training_strategy_selector,
            rectifier=args.rectifier,
            sims_per_entry=args.sims_per_entry,
            number_policies_selected=args.number_policies_selected,
            meta_strategy_method=args.meta_strategy_method,
            sample_from_marginals=sample_from_marginals,
            symmetric_game=args.symmetric_game,
        )

        start_time = time.time()
        self.total_steps = [0]
        expl_check_step_count = 0
        gpsro_iteration = 0
        computed_safety_expl = False
        while True:
            # if args.verbose:
            print("\n\nIteration : {}".format(gpsro_iteration))
            g_psro_solver.iteration()
            meta_game = g_psro_solver.get_meta_game()
            self.meta_probabilities = g_psro_solver.get_meta_strategies()
            self.policies = g_psro_solver.get_policies()
            num_oracle_steps = g_psro_solver.get_num_oracle_steps()
            num_meta_game_steps = g_psro_solver.get_num_meta_game_steps()
            self.total_steps.append(num_meta_game_steps + num_oracle_steps)
            print("Time so far: {}".format(time.time() - start_time))
            print(f"total steps: {num_meta_game_steps+ num_oracle_steps}")
            print(
                f"fps: {(num_meta_game_steps + num_meta_game_steps) / (time.time() - start_time)}"
            )


            if self.total_steps[-1] > self.meta_args.max_steps:
                expl_agents = self.wrap_rl_agent(
                    self.meta_args.experiment_dir + f"/agent.pkl"
                )

                if self.meta_args.compute_exploitability:
                    models = [ag.get_model() for ag in expl_agents]
                    action_selection = ["sto" if args.oracle_type == "ppo" else "det"] * 2
                    self.expl_callback(
                        models[0],
                        models[1],
                        self.total_steps[-1],
                        action_selection=action_selection,
                    )
                break
            if expl_check_step_count > self.meta_args.compute_exploitability_every and self.meta_args.compute_exploitability:
                expl_agents = self.wrap_rl_agent(
                    self.meta_args.experiment_dir
                    + f"/agent.pkl"
                )
                if self.meta_args.compute_exploitability:
                    models = [ag.get_model() for ag in expl_agents]
                    action_selection = ["sto" if args.oracle_type == "ppo" else "det"] * 2
                    self.expl_callback(
                        models[0],
                        models[1],
                        self.total_steps[-1],
                        action_selection=action_selection,
                    )
                expl_check_step_count = 0

            expl_check_step_count += self.total_steps[-1] - self.total_steps[-2]
            gpsro_iteration += 1

        # self.policies = policies
        # self.meta_probabilities = meta_probabilities
        return self.save_psro_policies(
            g_psro_solver, self.meta_args.experiment_dir, gpsro_iteration + 1
        )

    def current_step(self):
        return self.total_steps[-1]

    def save_psro_policies(self, psro_solver, ckpt_dir, ckpt_idx):
        policies = psro_solver._policies
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        pol_id = len(policies)
        player0s = policies[0]
        player1s = policies[1]
        # import pdb; pdb.set_trace()
        player0s[-1]._policy.save(ckpt_dir + f"/policy0_ckpt{ckpt_idx}.pt")
        player1s[-1]._policy.save(ckpt_dir + f"/policy1_ckpt{ckpt_idx}.pt")
        meta_game = psro_solver.get_meta_game()
        np.save(ckpt_dir + f"/meta_game_{ckpt_idx}.npy", meta_game)
        meta_strategies = psro_solver.get_meta_strategies()
        np.save(ckpt_dir + f"/meta_strategies_{ckpt_idx}.npy", meta_strategies)
        return ckpt_dir, ckpt_dir + f"/meta_strategies_{ckpt_idx}.npy"

    def wrap_rl_agent(self, save_path=None):
        n_obs = self.game.observation_tensor_size()
        n_actions = self.game.num_distinct_actions()

        # construct the policies from the q network of each psro subagent
        # this should be in the form of a list of lists, where each elemebnt is a torch neural networks
        agent_type = self.args.oracle_type
        policy_networks = [[], []]

        if agent_type == "dqn":
            for agent in range(2):
                for strategies in self.policies[agent]:
                    policy_networks[agent].append(strategies._policy._q_network)
        elif agent_type == "ppo":
            for agent in range(2):
                for strategies in self.policies[agent]:
                    policy_networks[agent].append(
                        strategies._policy.agent.network.actor
                    )

        meta_strategies = np.array(self.meta_probabilities)

        agents = [
            PSRORLAgent(
                agent_type=agent_type,
                models=policy_networks[i],
                strategy_profile=meta_strategies[i],
                player_id=i,
                n_obs=n_obs,
                n_actions=n_actions,
            )
            for i in range(2)
        ]

        if save_path:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(agents, f)

        return agents
