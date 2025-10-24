from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="iig_rl_benchmark",
    version="1.0.0",
    description="IIG-RL-Benchmark is a library for training and evaluating game theoretical or deep RL algorithms on OpenSpiel games.",
    author="Max Rudolph, Nathan Lichtlé, Sobhan Mohammadpour, Alexandre Bayen, J. Zico Kolter, Amy Zhang, Gabriele Farina, Eugene Vinitsky, and Samuel Sokota",
    packages=find_packages(exclude=["tests", "outputs", "results"]),
    install_requires=requirements,
    python_requires=">=3.8",
)
