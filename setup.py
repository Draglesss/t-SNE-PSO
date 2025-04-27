import os
import re

from setuptools import setup

# Read version from _version.py
with open(os.path.join("tsne_pso", "_version.py"), "r") as f:
    version_file = f.read()
    version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", version_file)
    if version_match:
        version = version_match.group(1)
    else:
        version = "0.0.1"

setup(
    name="tsne_pso",
    version=version,
    description="t-Distributed Stochastic Neighbor Embedding with Particle Swarm Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author=(
        "Allaoui, Mebarka; Belhaouari, Samir Brahim; Hedjam, Rachid; "
        "Bouanane, Khadra; Kherfi, Mohammed Lamine"
    ),
    maintainer="Otmane Fatteh",
    maintainer_email="fattehotmane@hotmail.com",
    url="https://github.com/draglesss/t-SNE-PSO",
    license="BSD-3-Clause",
    packages=["tsne_pso"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.3",
        "tqdm>=4.64.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
