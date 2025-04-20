from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tsne_pso",
    version="1.1.5",
    author="Otmane Fatteh",
    author_email="fattehotmane@hotmail.com",
    description="t-Distributed Stochastic Neighbor Embedding with Particle Swarm Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/draglesss/t-SNE-PSO",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
        "umap-learn>=0.5.3",
        "tqdm>=4.64.0",
    ],
    license="BSD-3-Clause",
)
