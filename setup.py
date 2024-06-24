from setuptools import setup, find_packages

setup(
    name='enf-jax',
    version='0.1.0',
    author='David Wessels, David Knigge',
    author_email='davidwessels15@gmail.com',
    description='A Jax implementation of the equivariant neural fields (ENF).',
    url='https://github.com/dafidofff/enf-jax',
    packages=find_packages(exclude=['experiments']),
)