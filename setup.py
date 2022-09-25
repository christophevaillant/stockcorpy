from setuptools import setup, find_packages

setup(
    name="stockcorpy",
    version="0.0.1",
    packages=find_packages(include=["stockcorpy"]),
    install_requires=[
        "numpy>=1.23.0",
        "pycoingecko>=3.0.0",
        "matplotlib>=3.5.0",
        ]
    )
