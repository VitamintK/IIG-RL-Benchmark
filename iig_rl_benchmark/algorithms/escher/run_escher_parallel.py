import torch
import open_spiel.python.rl_agent as rl_agent
from algorithms.escher.escher_parallel import ESCHERSolver
from torch.distributions import Categorical
import numpy as np

# from onnx2pytorch import ConvertModel
import torch.nn as nn
import ray
import time


class EscherRLAgent(rl_agent.AbstractAgent):
    def __init__(self, policy_network, player_id, n_obs, n_actions):
        self.model = policy_network
        self.model.to("cpu")
        self.player_id = player_id
        self.n_obs = n_obs
        self.n_actions = n_actions

    def step(self, time_step, is_evaluation=False):
        obs = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        legal_actions = np.array(legal_actions)
        legal_actions_mask = torch.zeros((self.n_actions,), dtype=torch.bool)
        legal_actions_mask[legal_actions] = True
        logits = self.model(torch.tensor(np.array(obs), dtype=torch.float))
        logits = torch.where(legal_actions_mask.bool(), logits, -1e6)
        dist = Categorical(logits=logits)
        action = dist.sample().detach().numpy()
        probs = np.zeros(self.n_actions)
        probs[action] = 1.0
        return rl_agent.StepOutput(action=action, probs=probs)

    def get_model(self):
        def model(info_state):
            action_values = self.model(torch.tensor(info_state, dtype=torch.float32))
            return action_values

        return model


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
        weights = params[::2]
        biases = params[1::2]
        param_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # need to transpose the weights
                layer.weight.data = torch.tensor(weights[param_idx]).T
                layer.bias.data = torch.tensor(biases[param_idx])
                param_idx += 1


def convert_tf_list_to_torch(params):
    layer_shapes = [layer_param.shape for layer_param in params][::2]
    torch_mlp = MLP(
        layer_shapes[0][0],
        [shape[1] for shape in layer_shapes[:-1]],
        layer_shapes[-1][1],
    )
    torch_mlp.load_params_from_dict(params)
    return torch_mlp


class RunEscherParallel:
    def __init__(self, args, game, expl_callback):
        self.args = args
        self.game = game
        self.expl_callback = expl_callback
        self.deep_cfr_solver = ray.remote(num_cpus=args.algorithm.num_cpus, num_gpus=0)(
            ESCHERSolver
        ).remote(
            self.game,
            num_traversals=args.algorithm.num_traversals,
            num_val_fn_traversals=args.algorithm.num_val_fn_traversals,
            num_iterations=args.algorithm.iters,
            policy_network_layers=args.algorithm.policy_network_layers,
            value_network_layers=args.algorithm.value_network_layers,
            regret_network_layers=args.algorithm.regret_network_layers,
            regret_network_train_steps=args.algorithm.regret_train_steps,
            policy_network_train_steps=args.algorithm.policy_net_train_steps,
            batch_size_regret=args.algorithm.batch_size_regret,
            value_network_train_steps=args.algorithm.val_train_steps,
            batch_size_value=args.algorithm.batch_size_val,
            train_device=args.algorithm.train_device,
            learning_rate=args.algorithm.learning_rate,
            expl=args.algorithm.expl,#: float = 1.0,
            val_expl=args.algorithm.val_expl,#: float = 0.01,
            # num_random_games=args.algorithm.num_random_games,
            max_steps=args.max_steps,
            use_wandb=False,#args.wandb,
            eval_every=args.algorithm.eval_every,
            create_worker_group=True,
            num_experience_workers=args.algorithm.num_workers,
            num_cpus=args.algorithm.num_cpus,
        )

    def wrap_rl_agent(self, save_path=None):
        policy_network_weights = ray.get(
            self.deep_cfr_solver.get_policy_weights.remote()
        )
        model = convert_tf_list_to_torch(policy_network_weights)
        n_obs = self.game.observation_tensor_size()
        n_actions = self.game.num_distinct_actions()
        agents = [
            EscherRLAgent(model, id, n_obs, n_actions)
            for id in range(self.game.num_players())
        ]
        if save_path:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(agents, f)
        return agents

    def run(self):
        ray.get(self.deep_cfr_solver.pre_iteration.remote())
        expl_check_step_count = 0
        wandb_log = {}
        computed_safety_expl = False
        self.current_nodes = 0
        while True:
            start_nodes = wandb_log.get("global_step", 0)
            wandb_log = ray.get(self.deep_cfr_solver.iteration.remote())
            # wandb.log(wandb_log)
            self.current_nodes = wandb_log.get("global_step", 0)

            added_nodes = self.current_nodes - start_nodes

            wandb_log["regret_loss_0"] = wandb_log["regret_loss"][0][0].numpy()
            wandb_log["regret_loss_1"] = wandb_log["regret_loss"][1][0].numpy()
            wandb_log["value_loss"] = wandb_log["value_loss"].numpy()
            del wandb_log["regret_loss"]
            expl_check_step_count += added_nodes

            if self.current_nodes >= self.args.max_steps:
                ray.get(self.deep_cfr_solver._reinitialize_policy_network.remote())
                ray.get(self.deep_cfr_solver._learn_average_policy_network.remote())
                expl_agents = self.wrap_rl_agent(self.args.experiment_dir + "/agent.pkl")
                if self.args.compute_exploitability:
                    models = [ag.get_model() for ag in expl_agents]
                    self.expl_callback(models[0], models[1], self.current_nodes)
                break

            if expl_check_step_count > self.args.compute_exploitability_every:
                ray.get(self.deep_cfr_solver._reinitialize_policy_network.remote())
                ray.get(self.deep_cfr_solver._learn_average_policy_network.remote())
                expl_agents = self.wrap_rl_agent()
                models = [ag.get_model() for ag in expl_agents]
                self.wrap_rl_agent(
                    self.args.experiment_dir + f"/agent.pkl"
                )
                if self.args.compute_exploitability:
                    self.expl_callback(models[0], models[1], self.current_nodes)
                expl_check_step_count = 0

        ray.shutdown()
        # last_iterate_policy = self.deep_cfr_solver.save_policy_network(self.args.experiment_dir)

    def current_step(self):
        return self.current_nodes
