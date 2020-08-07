from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split()

setup(name="dda",
      version="0.0.2",
      author="moskomule",
      packages=find_packages(include=['dda']),
      url="https://github.com/moskomule/dda",
      description="Differentiable Data Augmentation Library",
      long_description=readme,
      license="MIT",
      install_requires=requirements,
      )
