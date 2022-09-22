from setuptools import setup


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='imagerec',
      version='1.0',
      description='Image recognition for MDP',
      author='Group 27',
      packages=['imagerec'],
      install_requires=required
     )
