import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blochsim",
    version="0.0.1",
    author="Kwang Eun Jang",
    author_email="ke.jang@gmail.com",
    description="Python Bloch Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kejang/blochsim",
    project_urls={
        "Bug Tracker": "https://github.com/kejang/blochsim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
    ],
)
