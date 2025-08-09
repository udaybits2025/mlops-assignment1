from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = "California housing",
    version = "0.1",
    author = "BITS Group 138 ",
    packages = find_packages(),
    install_requires = requirements,

)