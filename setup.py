from setuptools import setup

setup(
    name="IIG-RL-Benchmark",
    version="1.0.0",
    description="IIG-RL-Benchmark is a library for training and evaluating game theoretical or deep RL algorithms on OpenSpiel games.",
    author="Max Rudolph, Nathan Lichtlé, Sobhan Mohammadpour, Alexandre Bayen, J. Zico Kolter, Amy Zhang, Gabriele Farina, Eugene Vinitsky, and Samuel Sokota",
    packages=["iig_rl_benchmark"],
    package_dir={"iig_rl_benchmark": "algorithms"},
)
