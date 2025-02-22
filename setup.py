import setuptools
import os
from pathlib import Path
try:
    from importlib.metadata import version  # Python 3.8+
except ImportError:
    from importlib_metadata import version  # For older Python versions

module_path = Path(os.path.abspath(__file__)).parent.absolute()

package_name = "blochsim"
try:
    pkg_version = version(package_name)
except Exception:
    pkg_version = "0.1.6"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=package_name,
    version=pkg_version,
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
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
)