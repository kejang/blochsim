import setuptools
import os
from pathlib import Path

module_path = Path(os.path.abspath(__file__)).parent.absolute()

ver = {}
with open(module_path.joinpath('version.py')) as ver_file:
    exec(ver_file.read(), ver)

module_path = Path(os.path.abspath(__file__)).parent.absolute()
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blochsim",
    version=ver['__version__'],
    author="Kwang Eun Jang",
    author_email="kejang@stanford.edu",
    description="Python Bloch Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/blochsim",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/blochsim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
    ],
)
