import setuptools

requirements = [
    "nnAudio",
    "librosa",
    "resampy",
    "imageio",
    "Pillow",
    "matplotlib",
    "seaborn",
    "click",
    "loguru",
    "test-tube",
    "colorama",
    "pyfiglet",
    "tqdm",
    "joblib",
    "natsort",
    "addict",
    "pyyaml",
    "cached-property",
    "boltons",
    "deprecation",
    "numba==0.48.0",
]

dev_requirements = [
    "black",
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"dev": dev_requirements,},
    entry_points={"console_scripts": ["beatbrain=beatbrain.cli:main"]},
)
