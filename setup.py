import setuptools

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

dev_requirements = [
    "black",
    "pytest",
    "jupyter",
    "pytest>=5.0.0",
    "coverage",
    "pytest-cov",
    "codecov",
    "pytest-dotenv",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beatbrain",
    version="1.0.0",
    author="Krishna Penukonda",
    url="https://github.com/tasercake/beatbrain",
    long_description=long_description,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["*.yaml", "*.yml"]},  # Include builtin configs
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={"dev": dev_requirements,},
    entry_points={"console_scripts": ["beatbrain=beatbrain.__main__:main"]},
)
