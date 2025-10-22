"""Phasic Policy Gradient (PPG)
Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py
"""

# pylint: disable=g-importing-member
import random
import time
import numpy as np
import torch
import psutil
import os

import pyspiel
from algorithms.ppg.ppg import PPG
from algorithms.ppg.ppg import PPGAgent
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.rl_environment import Environment
from open_spiel.python.vector_env import SyncVectorEnv
import open_spiel.python.rl_agent as rl_agent


def make_single_env(game_name, seed, config):
    def gen_env():
        game = pyspiel.load_game(game_name)
        return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

    return gen_env


class RunPPG:
    def __init__(self, config, game, expl_callback):
        self.config = config.algorithm
        self.meta_config = config
        self.game = game
        self.expl_callback = expl_callback

    def run(self):
        batch_size = int(self.config.num_envs * self.config.num_steps)

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )

        envs = SyncVectorEnv(
            [
                make_single_env(
                    str(self.game),
                    self.meta_config.seed + i,
                    self.meta_config,
                )()
                for i in range(self.config.num_envs)
            ]
        )
        self.agent_fn = PPGAgent

        game = envs.envs[0]._game  # pylint: disable=protected-access
        num_players = game.num_players()
        info_state_shape = game.information_state_tensor_shape()

        assert num_players == 1 or (
            num_players == 2
            and game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
        )
        assert envs.envs[0].is_turn_based
        assert game.get_type().reward_model == pyspiel.GameType.RewardModel.TERMINAL

        self.agent = PPG(
            input_shape=info_state_shape,
            num_actions=game.num_distinct_actions(),
            num_players=game.num_players(),
            num_envs=self.config.num_envs,
            steps_per_batch=self.config.num_steps,
            num_minibatches=self.config.num_minibatches,
            update_epochs=self.config.update_epochs,
            learning_rate=self.config.learning_rate,
            gae=self.config.gae,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=self.config.norm_adv,
            clip_coef=self.config.clip_coef,
            clip_vloss=self.config.clip_vloss,
            entropy_coef=self.config.ent_coef,
            value_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            target_kl=self.config.target_kl,
            n_iteration=self.config.n_iteration,
            e_policy=self.config.e_policy,
            v_value=self.config.v_value,
            e_auxiliary=self.config.e_auxiliary,
            beta_clone=self.config.beta_clone,
            num_aux_rollouts=self.config.num_aux_rollouts,
            n_aux_grad_accum=self.config.n_aux_grad_accum,
            device=device,
            agent_fn=self.agent_fn,
            log_file=os.path.join(self.meta_config.experiment_dir, 'train_log.csv'),
        )

        time_steps = envs.reset()
        cp_step = 0
        t0 = time.time()

        num_updates = self.meta_config.max_steps // batch_size + 1
        num_phases = num_updates // self.config.n_iteration
        updates_count = -1

        phase = -1
        computed_safety_expl = False
        while self.agent.total_steps_done < self.meta_config.max_steps:
            phase += 1
            # POLICY PHASE
            for update in range(self.config.n_iteration):
                updates_count += 1
                for _ in range(self.config.num_steps):
                    # Output of current player in each of the envs
                    agent_outputs = self.agent.step(time_steps)

                    # Advance all envs
                    time_steps, rewards, dones, unreset_time_steps = envs.step(
                        agent_outputs, reset_if_done=True
                    )
                    self.agent.post_step([reward[0] for reward in rewards], dones)

                if self.config.anneal_lr:
                    self.agent.anneal_learning_rate(updates_count, num_updates)
                self.agent.learn(time_steps, update)

                if self.agent.total_steps_done > cp_step + self.meta_config.compute_exploitability_every:
                    cp_step = cp_step + self.meta_config.compute_exploitability_every
                    if self.expl_callback is not None:
                        self.expl_callback(
                            self.get_model(),
                            self.get_model(),
                            self.agent.total_steps_done,
                        )
                    self.agent.save(f"{self.meta_config.experiment_dir}/agent.pth")

                if updates_count % self.config.eval_every == 0:
                    time_elapsed = time.time() - t0
                    time_remaining_est = (
                        (self.meta_config.max_steps - self.agent.total_steps_done)
                        * time_elapsed
                        / self.agent.total_steps_done
                    )
                    print(
                        f"step {self.agent.total_steps_done}/{self.meta_config.max_steps} ; elapsed: {time_elapsed/60:.1f}min ; remaining: {time_remaining_est/60:.1f}min"
                    )

            # AUXILIARY PHASE
            self.agent.learn_auxiliary()

        if self.expl_callback is not None:
            self.expl_callback(
                self.get_model(), self.get_model(), self.agent.total_steps_done
            )

        self.agent.save(f"{self.meta_config.experiment_dir}/agent.pth")

        self.network = self.agent.network

    def current_step(self):
        return self.agent.total_steps_done

    def load_cp(self, cp_path):
        print("loading checkpoint from", cp_path)

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.cuda else "cpu"
        )

        self.network = PPGAgent(
            num_actions=self.game.num_distinct_actions(),
            observation_shape=self.game.information_state_tensor_shape(),
            device=device,
        ).to(device)

        self.network.actor.load_state_dict(torch.load(cp_path))

    def wrap_rl_agent(self, *args, **kwargs):
        class PPGRLAgent(rl_agent.AbstractAgent):
            def __init__(self, model, player_id, n_actions):
                self.model = model
                # self.model.to('cpu')
                self.player_id = player_id
                self.n_actions = n_actions

            def step(self, time_step, is_evaluation=False):
                obs = time_step.observations["info_state"][self.player_id]
                legal_actions = time_step.observations["legal_actions"][self.player_id]
                legal_actions = np.array(legal_actions)
                legal_actions_mask = torch.zeros((self.n_actions,), dtype=torch.bool)
                legal_actions_mask[legal_actions] = True
                action = self.model.get_action_and_value(
                    x=torch.tensor(np.array(obs), dtype=torch.float),
                    legal_actions_mask=legal_actions_mask,
                )[0]
                return rl_agent.StepOutput(action=action, probs=None)

            def get_model(self):
                return self.model.actor

        return [
            PPGRLAgent(
                self.network, player_id, n_actions=self.game.num_distinct_actions()
            )
            for player_id in range(self.game.num_players())
        ]

    def get_model(self):
        return self.agent.network.actor
