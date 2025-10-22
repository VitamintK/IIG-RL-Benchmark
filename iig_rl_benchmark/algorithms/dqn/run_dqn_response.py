"""RL DQN best response (from OpenSpiel's rl_response.py)

RL agents trained against fixed policy/bot as approximate responses.

This can be used to try to find exploits in policies or bots, as described in
Timbers et al. '20 (https://arxiv.org/abs/2004.09677), but only using RL
directly rather than RL+Search.
"""

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

from eval.head2head_eval import eval_against_fixed_bots
import wandb
import sys
import time
from argparse import ArgumentParser
import numpy as np
import torch
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import random_agent
from open_spiel.python.rl_environment import Environment


sys.path.append("..")
sys.path.append(".")


def get_dqn_model(dqn):
    def model(info_state):
        action_values = dqn._q_network(info_state)
        return action_values

    return model


def create_br_agents(args, num_players, info_state_size, num_actions):
    if args.dqn_best_responder == "tf":
        from open_spiel.python.algorithms import dqn
        import tensorflow.compat.v1 as tf

        sess = tf.Session()
        learning_agents = [
            dqn.DQN(
                sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                discount_factor=args.dqn_discount_factor,
                epsilon_start=args.dqn_epsilon_start,
                epsilon_end=args.dqn_epsilon_end,
                hidden_layers_sizes=[int(ls) for ls in args.dqn_hidden_layers_sizes],
                replay_buffer_capacity=args.dqn_replay_buffer_capacity,
                batch_size=args.dqn_batch_size,
                epsilon_decay_duration=args.dqn_epsilon_decay_duration,
                optimizer_str=args.dqn_optimizer_str,
                update_target_network_every=args.dqn_update_target_network_every,
            )
            for idx in range(num_players)
        ]
    elif args.dqn_best_responder == "torch":
        from open_spiel.python.pytorch import dqn

        dqn.ILLEGAL_ACTION_LOGITS_PENALTY = -1e9  # fix a bug in the DQN implementation
        learning_agents = [
            dqn.DQN(
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                discount_factor=args.dqn_discount_factor,
                epsilon_start=args.dqn_epsilon_start,
                epsilon_end=args.dqn_epsilon_end,
                hidden_layers_sizes=[int(ls) for ls in args.dqn_hidden_layers_sizes],
                replay_buffer_capacity=args.dqn_replay_buffer_capacity,
                batch_size=args.dqn_batch_size,
                epsilon_decay_duration=args.dqn_epsilon_decay_duration,
                optimizer_str=args.dqn_optimizer_str,
                update_target_network_every=args.dqn_update_target_network_every,
            )
            for idx in range(num_players)
        ]
    elif args.dqn_best_responder == "uniform":
        learning_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

    return learning_agents


class DQNRLAgent(rl_agent.AbstractAgent):
    def __init__(
        self,
        model,
        player_id,
        n_obs,
        n_actions,
    ):
        self.player_id = player_id
        self.num_actions = n_actions
        self.num_obs = n_obs
        self.model = model

    def step(self, time_step, is_evaluation=False):
        if time_step.last():
            return
        obs = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        legal_actions = np.array(legal_actions)
        legal_actions_mask = torch.zeros((self.num_actions,), dtype=torch.bool)
        legal_actions_mask[legal_actions] = True
        logits = self.model(torch.tensor(np.array(obs), dtype=torch.float))
        logits = torch.where(legal_actions_mask.bool(), logits, -1e9)
        action = torch.argmax(logits)
        probs = np.zeros(self.num_actions)
        probs[action] = 1.0
        return rl_agent.StepOutput(action=int(action), probs=probs)

    def get_model(
        self,
    ):
        return self.model


def run_dqn_response(args, game, rl_agents=None):
    # setup logging
    log_path = args.experiment_dir
    print(f"Logging at {log_path}")

    env = Environment(game)

    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    assert rl_agents is not None, "rl_agents must be provided"
    exploitee_agents = rl_agents
    num_players = len(exploitee_agents)

    learning_agents = create_br_agents(args, num_players, info_state_size, num_actions)

    global_step = 0
    iteration = 0
    total_time_eval = 0
    total_time_step = 0
    total_time_learn = 0
    time_start = time.time()
    print(f"DQN: step 0/{args.dqn_max_steps}")
    while global_step < args.dqn_max_steps:
        # eval
        if iteration % args.dqn_eval_every == 0:
            t0 = time.time()
            episode_rewards_br_exp = eval_against_fixed_bots(
                game, learning_agents[0], exploitee_agents[1], args.dqn_eval_episodes
            )
            episode_rewards_exp_br = eval_against_fixed_bots(
                game, exploitee_agents[0], learning_agents[1], args.dqn_eval_episodes
            )

            r_mean_0 = np.mean(episode_rewards_br_exp[0])
            r_mean_1 = np.mean(episode_rewards_exp_br[1])

            r_mean_avg = (r_mean_0 + r_mean_1) / 2

            total_time_eval += time.time() - t0
            if args.wandb:
                wandb.log(
                    {
                        "eval/r_mean_0": r_mean_0,
                        "eval/r_mean_1": r_mean_1,
                        "eval/r_mean_avg": r_mean_avg,
                        "eval/iter": iteration,
                        "eval/global_step": global_step,
                    }
                )

            print(
                f"step {str(global_step):8s}/{args.dqn_max_steps} ({iteration=}): {r_mean_0=}, {r_mean_1=}, {r_mean_avg=}"
            )

            if iteration > 0:
                time_elapsed = time.time() - time_start
                time_remaining_est = (
                    (args.dqn_max_steps - global_step) * time_elapsed / global_step
                )
                proportion_eval = total_time_eval / time_elapsed * 100
                proportion_step = total_time_step / time_elapsed * 100
                proportion_learn = total_time_learn / time_elapsed * 100
                print(
                    f"remaining: {time_remaining_est/60:.1f}min (elapsed: {time_elapsed/60:.1f}min) ; SPS: {global_step/time_elapsed:.1f} ; proportions: {proportion_step:.1f}% agent step, {proportion_learn:.1f}% dqn learn, {proportion_eval:.1f}% eval"
                )

        # collect data for two games (and train)
        agents_round1 = [learning_agents[0], exploitee_agents[1]]
        agents_round2 = [exploitee_agents[0], learning_agents[1]]

        for agents in [agents_round1, agents_round2]:
            t0 = time.time()
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                else:
                    agents_output = [agent.step(time_step) for agent in agents]
                    action_list = [
                        agent_output.action for agent_output in agents_output
                    ]
                time_step = env.step(action_list)
                global_step += 1
            total_time_step += time.time() - t0

            # Episode is over, step all agents with final info state.
            t0 = time.time()
            for agent in agents:
                agent.step(time_step)
            total_time_learn += time.time() - t0

        # increment
        iteration += 1
    for i, agent in enumerate(learning_agents):
        agent.save(log_path + f"/final_exploiter_player_{i}.pt")
    # return learning_agents
    return [
        DQNRLAgent(get_dqn_model(agent), i, info_state_size, num_actions)
        for i, agent in enumerate(learning_agents)
    ]


if __name__ == "__main__":
    # parse cli args
    parser = ArgumentParser()
    args = parser.parse_args()
    run_dqn_response(args)
