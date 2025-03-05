from setuptools import setup, find_packages

setup(
    name="lensingpop",
    description="Testing population assumptions in lensing",
    url="https://github.com/sterinaldi/lensingpop",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        'numpy', 'lalsuite', 'dill', 'matplotlib', 'figaro', 'scipy', 'tqdm', 
        'pathlib', 'corner', 'numba', 'cython', 'setuptools_scm'
    ],
    version="1.0.0"
)
