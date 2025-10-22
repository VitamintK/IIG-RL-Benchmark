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
"""Tests for RNaD algorithm under open_spiel."""

import jaxlib
import numpy as np

# from open_spiel.python.algorithms.rnad import rnad
import sys

sys.path.append(".")
from algorithms.rnad import rnad
from copy import deepcopy
import torch
import torch.nn as nn
import time
from open_spiel.python import rl_agent
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--num_iterations", type=int, default=1000009, help="num_iteration to run rnad"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="directory to save results"
    )
    parser.add_argument(
        "--wandb", type=bool, default=False, help="whether to log to wandb"
    )
    parser.add_argument(
        "--game", type=str, default="phantom_ttt", help="game to run rnad on"
    )

    return parser.parse_args()


def convert_jax_to_list(obj):
    if isinstance(obj, jaxlib.xla_extension.ArrayImpl):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_jax_to_list(v) for k, v in obj.items()}
    return obj


def convert_dict_to_torch(params):
    layer_params = list(params.values())[:-1]
    layer_shapes = [np.array(layer_param["w"]).shape for layer_param in layer_params]
    torch_mlp = MLP(
        layer_shapes[0][0],
        [shape[1] for shape in layer_shapes[:-1]],
        layer_shapes[-1][1],
    )
    torch_mlp.load_params_from_dict(layer_params)
    return torch_mlp


def unit_test_jax_to_torch(solver, model):
    jax_pi = np.array(
        solver._network_jit_apply(solver.params, solver.test_env_step).tolist()
    )
    torch_test_tensor_obs = torch.tensor(solver.test_env_step.obs, dtype=torch.float32)
    torch_logits = model(torch_test_tensor_obs)
    torch_pi = torch.nn.functional.softmax(torch_logits, dim=-1).detach().numpy()
    print("Testing JAX to PyTorch conversion...")
    pi_diff_mean = np.max(jax_pi[0] - torch_pi[0])
    assert (
        pi_diff_mean < 1e-4
    ), f"JAX to PyTorch conversion failed. Mean difference: {pi_diff_mean}, \n jax: {jax_pi[0]} \n torch {torch_pi[0]} \n obs: {torch_test_tensor_obs[0]}"


def save_jax_model_as_pytorch(solver, path=None):
    params = convert_jax_to_list(deepcopy(solver.__getstate__()["params"]))
    torch_model = convert_dict_to_torch(params)
    if path is not None:
        torch.save(torch_model, path)
    # unit_test_jax_to_torch(solver, torch_model)
    return torch_model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        # Create a list of all layers
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load_params_from_dict(self, params):
        param_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # need to transpose the weights
                layer.weight.data = torch.tensor(params[param_idx]["w"]).T
                layer.bias.data = torch.tensor(params[param_idx]["b"])
                param_idx += 1


class RNaDRLAgent(rl_agent.AbstractAgent):
    def __init__(self, model, player_id, n_obs, n_actions):
        self.model = model
        self.model.to("cpu")
        self.player_id = player_id
        self.n_obs = n_obs
        self.n_actions = n_actions

    def step(self, time_step, is_evaluation=False):
        assert False, "Not implemented"
        pass

    def get_model(self):
        def model(info_state):
            action_values = self.model(torch.tensor(info_state, dtype=torch.float32))
            return action_values

        return model


class RunRNaD:
    def __init__(self, args, game, expl_callback):
        self.args = args.algorithm
        self.meta_args = args
        self.game = game
        self.expl_callback = expl_callback
        self.args.seed = self.meta_args.seed

    def wrap_rl_agent(self, save_path=None):
        n_obs = self.game.observation_tensor_size()
        n_actions = self.game.num_distinct_actions()
        agents = [
            RNaDRLAgent(self.cur_model, id, n_obs, n_actions)
            for id in range(self.game.num_players())
        ]
        if save_path:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(agents, f)
        return agents

    def run(self):
        rnad_solver = rnad.RNaDSolver(self.args, self.game)
        self.model = save_jax_model_as_pytorch(rnad_solver, "rnad_params_init.pth")
        expl_check_step_count = 0
        self.step_count = 0
        start_time = time.time()
        computed_safety_expl = False
        while True:
            logs = rnad_solver.step()
            logs["loss"] = logs["loss"].item()
            add_steps = rnad_solver.actor_steps - self.step_count
            self.step_count = rnad_solver.actor_steps
            logs["global_step"] = self.step_count
            logs["fps"] = self.step_count / (time.time() - start_time)
            self.cur_model = save_jax_model_as_pytorch(rnad_solver)
            expl_check_step_count += add_steps
            if expl_check_step_count > self.meta_args.compute_exploitability_every and self.step_count <= self.meta_args.max_steps:
                expl_agents = self.wrap_rl_agent(
                    self.meta_args.experiment_dir + f"/agent.pkl"
                )
                if self.meta_args.compute_exploitability:
                    models = [ag.get_model() for ag in expl_agents]
                    self.expl_callback(models[0], models[1], self.step_count)
                expl_check_step_count = 0

            print(f"steps completed: {self.step_count}")
            if self.step_count > self.meta_args.max_steps:
                expl_agents = self.wrap_rl_agent(
                    self.meta_args.experiment_dir + f"/agent.pkl"
                )
                if self.meta_args.compute_exploitability:
                    models = [ag.get_model() for ag in expl_agents]
                    self.expl_callback(models[0], models[1], self.step_count)
                break

    def current_step(self):
        return self.step_count


if __name__ == "__main__":
    args = get_args()
    runner = RunRNaD(args)
    runner.run()
