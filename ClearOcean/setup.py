from setuptools import setup, find_packages

setup(
    name='clearocean',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'basicsr>=1.4.2',
        'torch',
        'numpy',
        'tqdm',
    ],
)