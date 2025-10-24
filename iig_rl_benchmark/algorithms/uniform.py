import numpy as np
import pyspiel
from open_spiel.python import rl_agent


class Uniform:
    def __init__(self, config):
        self.config = config
        self.game = pyspiel.load_game(self.config.game)

    def wrap_rl_agent(self):
        class UniformAgent:
            def __init__(self, player_id, n_actions):
                self.player_id = player_id
                self.n_actions = n_actions

            def step(self, time_step, is_evaluation=False):
                legal_actions = time_step.observations["legal_actions"][self.player_id]
                if len(legal_actions) == 0:
                    return None
                action = np.random.choice(legal_actions)
                return rl_agent.StepOutput(action=action, probs=None)

        return [
            UniformAgent(player_id, n_actions=self.game.num_distinct_actions())
            for player_id in range(self.game.num_players())
        ]
