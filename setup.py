from setuptools import setup
from setuptools import find_packages

setup(
    name='OBoW',
    version='0.0.1',
    description='OBoW',
    author='Spyros Gidaris',
    packages=find_packages(),
    install_requires=["tqdm",
                      "numpy",
                      "torch",
                      "torchvision",
                      "Pillow"]
    )
