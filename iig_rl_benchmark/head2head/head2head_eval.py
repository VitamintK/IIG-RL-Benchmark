from algorithms.eas_exploitability import (
    build_traverser,
    compute_exploitability_cached,
)
import pyspiel

import argparse
import yaml
import pickle
import torch
import numpy as np
import time
from torch import nn
import os


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PGModel(torch.nn.Module):
    def __init__(self, observation_shape, num_actions):
        super(PGModel, self).__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )

    def forward(self, x):
        return self.model(x)


def eas_expl_cb(
    t,
    game,
    model_p0,
    model_p1,
    action_selection=["sto", "sto"],
    probs_0=None,
    probs_1=None,
):
    print("Running eas best response (this will take a few minutes)")
    ev0, expl0, expl1, cached_probs_0, cached_probs_1 = compute_exploitability_cached(
        model_p0,
        model_p1,
        traverser=t,
        batch_size=400_000,
        action_selection=action_selection,
        game_name=game,
        probs_0=probs_0,
        probs_1=probs_1,
    )
    return {
        "ev0": ev0,
        "expl0": expl0,
        "expl1": expl1,
        "probs_0": cached_probs_0,
        "probs_1": cached_probs_1,
    }


def load_models(agent_path_dict, game_str):
    agent_dict = {"player_0": [], "player_1": []}

    for agent_path in agent_path_dict:
        if agent_path["algo"] in ["psro", "escher", "escher", "nfsp", "rnad"]:
            with open(agent_path["path"], "rb") as f:
                agents = pickle.load(f)
            models = [ag.get_model() for ag in agents]
            agent_dict["player_0"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[0],
                    "agent": agents[0],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
            agent_dict["player_1"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[1],
                    "agent": agents[1],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
        else:
            with open(agent_path["path"], "rb") as f:
                model_dict = torch.load(f, weights_only=True)
            if '0.weight' in model_dict:
                model_dict = {f"model.{k}": v for k, v in model_dict.items()}
            elif '0.0.weight' in model_dict: # PPG
                model_dict = {f"model.{k[2:]}" if k.count('.') == 2 else 'model.' + k.replace('1', '6'): v for k, v in model_dict.items()}
            else:
                raise ValueError(f"Unrecognized model dict keys: {model_dict.keys()}")

            game = pyspiel.load_game(game_str)
            models = [PGModel(observation_shape=game.information_state_tensor_size(), num_actions=game.num_distinct_actions()) for _ in range(2)]
            for i in range(2):
                models[i].load_state_dict(model_dict)

            agent_dict["player_0"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[0],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )
            agent_dict["player_1"].append(
                {
                    "algo": agent_path["algo"],
                    "model": models[1],
                    "path": agent_path["path"],
                    "probs": None,
                }
            )

            # assert False, "fix hard code for phantom_ttt"
    return agent_dict


def split_array(num_agents, k, part):
    chunk_size = num_agents // k
    chunks = []

    X = np.arange(num_agents)
    Y = np.arange(num_agents)

    X, Y = np.meshgrid(X, Y)

    for i in range(0, num_agents, chunk_size):
        for j in range(0, num_agents, chunk_size):
            X_ind = X[i : i + chunk_size, j : j + chunk_size]
            Y_ind = Y[i : i + chunk_size, j : j + chunk_size]
            chunks.append((X_ind.flatten(), Y_ind.flatten()))
    return chunks[part]


def main(args):
    start_time = time.time()
    with open(args.agents_yaml, "r") as f:
        agents_yaml = yaml.load(f, Loader=yaml.FullLoader)

    game = agents_yaml["game"]

    game_to_osgame = {
        'classical_phantom_ttt': 'phantom_ttt(obstype=reveal-nothing)',
        'abrupt_phantom_ttt': 'phantom_ttt(obstype=reveal-nothing,gameversion=abrupt)',
        'classical_dark_hex': 'dark_hex(gameversion=cdh,board_size=3,obstype=reveal-nothing)',
        'abrupt_dark_hex': 'dark_hex(gameversion=adh,board_size=3,obstype=reveal-nothing)',
        'kuhn_poker': 'kuhn_poker(players=2)',
        'leduc_poker': 'leduc_poker(players=2)',
    }

    agent_dict = load_models(agents_yaml["agents"], game_to_osgame[game])

    t = build_traverser(game)

    head2head_results = (
        np.zeros((len(agent_dict["player_0"]), len(agent_dict["player_1"]))) - 2
    )
    num_runs = len(agent_dict["player_0"]) * len(agent_dict["player_1"])
    count = 0

    num_agents = len(agent_dict["player_0"])
    player_0_idx, player_1_idx = split_array(num_agents, args.disc, args.part)

    for i, j in zip(player_0_idx, player_1_idx):
        print(f"Running head2head for {i} vs {j}")
        print(f"{count} out of {num_runs} runs")

        model_p0 = agent_dict["player_0"][i]["model"]
        model_p1 = agent_dict["player_1"][j]["model"]

        player_0_algo = agent_dict["player_0"][i]["algo"]
        player_1_algo = agent_dict["player_1"][j]["algo"]

        player_0_probs = agent_dict["player_0"][i]["probs"]
        player_1_probs = agent_dict["player_1"][j]["probs"]

        player_0_action_selection = (
            "det" if player_0_algo in ["psro"] else "sto"
        )
        player_1_action_selection = (
            "det" if player_0_algo in ["psro"] else "sto"
        )

        action_selection = [player_0_action_selection, player_1_action_selection]

        expl_dict = eas_expl_cb(
            t,
            game,
            model_p0,
            model_p1,
            action_selection=action_selection,
            probs_0=player_0_probs,
            probs_1=player_1_probs,
        )
        head2head_results[i, j] = expl_dict["ev0"]

        if player_0_probs is None:
            agent_dict["player_0"][i]["probs"] = expl_dict["probs_0"]
        if player_1_probs is None:
            agent_dict["player_1"][j]["probs"] = expl_dict["probs_1"]

        with open(args.save_dir + f"/head2head_results_{count}.yaml", "w") as file:
            agents_yaml.update({"head2head_results": head2head_results.tolist()})
            yaml.dump(agents_yaml, file)

        count += 1


    with open(args.save_dir + f"/head2head_results_final.yaml", "w") as file:
        agents_yaml.update({"head2head_results": head2head_results.tolist()})
        yaml.dump(agents_yaml, file)

    np.save(args.save_dir + f"/head2head_results_final.npy", head2head_results)
    end_time = time.time()
    print(f"Head2head evaluation took {end_time - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-yaml", type=str, default=None, required=True)
    parser.add_argument("--save-dir", type=str, default=None, required=True)
    parser.add_argument("--disc", type=int, default=1)
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
