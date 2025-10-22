# IIG-RL-Benchmark

IIG-RL-Benchmark is a library for running game theoretical and deep RL algorithms on [OpenSpiel](https://github.com/google-deepmind/open_spiel) games. Furthermore, we compute exact exploitability using the [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) library which currently supports Phantom Tic-Tac-Toe and 3x3 Dark Hex, as well as their abrupt versions.

Paper: [arxiv.org/abs/2502.08938](https://arxiv.org/abs/2502.08938)

Play against RL: [Online demo](https://www.nathanlichtle.com/research/2p0s)

## Citation

```
@article{rudolph2025reevaluating,
  title={Reevaluating Policy Gradient Methods for Imperfect-Information Games},
  author={Rudolph, Max and Lichtlé, Nathan and Mohammadpour, Sobhan and Bayen, Alexandre and Kolter, J. Zico and Zhang, Amy and Farina, Gabriele and Vinitsky, Eugene and Sokota, Samuel},
  journal={arXiv preprint arXiv:2502.08938},
  year={2025}
}
```

## Installation

Ensure that your conda is not activated to prevent spawning dual virtual environments.

```bash
# Download this repository (IIG-RL-Benchmark) and install dependencies
git clone https://github.com/nathanlct/IIG-RL-Benchmark.git
cd IIG-RL-Benchmark

# Update the exp-a-spiel (eas) dependency
git submodule init
git submodule update

# This will install eas and its dependencies. Make sure to do this from the top level repo.
pixi install
pixi shell

# Lastly, install the other requirements
pip install -r requirements.txt
```

### Installing exp-a-spiel on MacOS

The exp-a-spiel dependency (which is only required to compute exact exploitability of policies) supports installation on Linux. To install on OSX, use the following branch in exp-a-spiel (which has not been extensively tested): `osx_install` ([link](https://github.com/gabrfarina/exp-a-spiel/tree/osx_install)).

### (Optional alternative) Installing exp-a-spiel via `pip`

```
cd IIG-RL-Benchmark
pip install ./eas
```

## Usage

Train a PPO agent on Phantom Tic-Tac-Toe for 10M steps with an entropy coefficient of 0.05 and no exploitability computation.

```bash
python main.py algorithm=ppo game=classical_phantom_ttt max_steps=10000000 algorithm.ent_coef=0.05 compute_exploitability=False
```

Train an NFSP agent on Abrupt Dark Hex 3 for 10M steps and compute exact exploitability every 1M steps.

```bash
python main.py algorithm=nfsp game=abrupt_dark_hex max_steps=10000000 compute_exploitability=True compute_exploitability_every=1000000
```

The following algorithms are implemented: `ppo`, `nfsp`, `ppg`, `mmd`, `rnad`, `escher`, `psro`.

Exact exploitability is supported for the following games: `classical_phantom_ttt`, `abrupt_phantom_ttt`, `classical_dark_hex`, `abrupt_dark_hex`.

See the `configs` folder for all available command-line arguments.

## Evaluating head2head 

- Input the paths and algorithm names into a `yaml` file like the example in `head2head/example.yaml`

```
python head2head/head2head_eval.py --agents-yaml path/to/example.yaml --save-dir <save-dir>
```

This command will kick off head2head evaluation for the agents listed in the eval `yaml` using the `eas` library. 

## Games

The following games are also available to play against our RL agents in this [online demo](https://www.nathanlichtle.com/research/2p0s).

### Phantom Tic-Tac-Toe (PTTT)

**Tic-Tac-Toe:** Players take turn placing their symbol (X or O) on a 3x3 grid. The goal is to form a horizontal, vertical, or diagonal line of your symbol. X plays first.

**Phantom:** The opponent's moves are hidden. Selecting an occupied cell reveals it, and the player's turn continues.

**Abrupt PTTT**: In the abrupt variant, if a player's move reveals an opponent's symbol, their turn ends.

### 3x3 Dark Hex (DH3)

**Hex:** Players take turns placing their stones (red or blue) on a 3x3 hexagonal board. The goal is to form a chain of adjacent stones connecting both sides of your color. Red plays first.

**Dark:** The opponent's moves are hidden. Selecting an occupied cell reveals it, and the player's turn continues.

**Abrupt DH3:** In the abrupt variant, if a player's move reveals an opponent's symbol, their turn ends.
