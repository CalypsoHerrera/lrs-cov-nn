import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    REQUIRED_PKGS = f.read().splitlines()

setuptools.setup(
    name="lsr",
    version="0.0.1",
    description="Code for https://arxiv.org/abs/1908.00461",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CalypsoHerrera/lrs-cov-nn",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PKGS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
