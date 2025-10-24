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

"""NFSP agents trained on <game>"""


# import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
# from run_dqn_response import run_dqn_response

# from open_spiel.python.algorithms import nfsp

import algorithms.nfsp.nfsp as nfsp
import wandb
from time import time
import numpy as np
import torch
import open_spiel.python.rl_agent as rl_agent
from algorithms.nfsp.nfsp import MODE


# return NFSP RL Agent class
class NFSPRLAgent(rl_agent.AbstractAgent):
    def __init__(
        self,
        model,
        player_id,
        n_obs,
        n_actions,
    ):
        self.player_id = player_id
        self.n_obs = n_obs
        self.n_actions = n_actions
        self._model = model

    def step(self, time_step, is_evaluation=False):
        # Act step: don't act at terminal info states.
        if time_step.last():
            return

        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]

        info_state = np.reshape(info_state, [1, -1])
        action_values = self._model(torch.as_tensor(info_state).to(torch.float32))

        action_probs = torch.softmax(action_values, dim=1)
        self._last_action_values = action_values[0]

        probs = np.zeros(self.n_actions)
        probs[legal_actions] = action_probs[0][legal_actions].detach().numpy()
        probs /= sum(probs)
        action = np.random.choice(len(probs), p=probs)

        return rl_agent.StepOutput(action=action, probs=probs)

    def get_model(self):
        return self._model


class NFSPPolicies(policy.Policy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None
        )

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


class RunNFSP:
    def __init__(self, args, game, expl_callback):
        self.args = args
        self.game = game
        self.expl_callback = expl_callback

    def run(self):
        args = self.args

        cktp_path = args.experiment_dir
        num_players = 2

        env = rl_environment.Environment(self.game)

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        hidden_layers_sizes = [int(l) for l in args.algorithm.hidden_layers_sizes]
        kwargs = {
            "reservoir_buffer_capacity": args.algorithm.reservoir_buffer_capacity,
            "min_buffer_size_to_learn": args.algorithm.min_buffer_size_to_learn,
            "anticipatory_param": args.algorithm.anticipatory_param,
            "batch_size": args.algorithm.batch_size,
            "learn_every": args.algorithm.learn_every,
            "sl_learning_rate": args.algorithm.sl_learning_rate,
            "optimizer_str": args.algorithm.optimizer_str,
            "loss_str": args.algorithm.loss_str,
            "inner_rl_agent_type": args.algorithm.inner_rl_agent.algorithm_name,
            "inner_rl_agent_args": args.algorithm.inner_rl_agent,
        }
        agents = [
            nfsp.NFSP(idx, info_state_size, num_actions, hidden_layers_sizes, **kwargs)
            for idx in range(num_players)
        ]
        self.agents = agents

        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
        old_time = time()
        self.step_count = 0
        expl_check_step_count = 0
        ep = 0
        computed_safety_expl = False
        while True:
            if self.step_count > args.max_steps:
                expl_agents = self.wrap_rl_agent(self.args.experiment_dir + "/agent.pkl")
                models = [ag.get_model() for ag in expl_agents]
                if self.expl_callback is not None:
                    self.expl_callback(models[0], models[1], self.step_count)
                break
            if (ep + 1) % args.algorithm.eval_every == 0:
                loss_type = ["sl_loss", "rl_loss"]
                losses = [agent.loss for agent in agents]
                wandb_losses = {
                    f"agent_{i}_{loss_type[j]}": losses[i][j]
                    for i in range(len(losses))
                    for j in range(len(loss_type))
                }
                wandb_losses["steps"] = self.step_count
                cur_time = time()
                wandb_losses["fps"] = args.algorithm.eval_every / (cur_time - old_time)
                old_time = cur_time
                print(wandb_losses)

            if expl_check_step_count > args.compute_exploitability_every:
                expl_agents = self.wrap_rl_agent(
                    self.args.experiment_dir + f"/agent.pkl"
                )
                models = [ag.get_model() for ag in expl_agents]
                if self.expl_callback is not None:
                    self.expl_callback(models[0], models[1], self.step_count)
                expl_check_step_count = 0
                for i, agent in enumerate(agents):
                    agent.save(cktp_path, ep)

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
                if (
                    args.algorithm.inner_rl_agent.algorithm_name == "ppo"
                    and agents[player_id]._mode == MODE.best_response
                ):
                    agents[player_id]._rl_agent.post_step(
                        time_step, is_evaluation=False
                    )
                self.step_count += 1
                expl_check_step_count += 1

            # Episode is over, step all agents with final info state.
            if (
                args.algorithm.inner_rl_agent.algorithm_name == "ppo"
                and agents[1 - player_id]._mode == MODE.best_response
            ):
                agents[1 - player_id]._rl_agent.post_step(
                    time_step, is_evaluation=False
                )
            if args.algorithm.inner_rl_agent.algorithm_name == "dqn":
                for agent in agents:
                    agent.step(time_step)

            for agent in agents:
                agent._sample_episode_policy()
            ep += 1

        path = agent.save(cktp_path, "final")
        self.expl_agents = expl_policies_avg
        self.game = env.game
        return path, agents

    def current_step(self):
        return self.step_count

    def wrap_rl_agent(self, save_path=None):
        n_obs = self.game.observation_tensor_size()
        n_actions = self.game.num_distinct_actions()
        agents = [
            NFSPRLAgent(self.agents[id]._avg_network, id, n_obs, n_actions)
            for id in range(self.game.num_players())
        ]

        if save_path:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(agents, f)

        return agents
