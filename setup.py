import setuptools

requirements = [
    "umap",
    "seaborn",
    "colorama",
    "tqdm",
    "IPython",
    "matplotlib",
    "joblib",
    "click",
    "numpy",
    "Pillow",
    "imageio",
    "natsort",
    "ffmpeg",
    "librosa",
    "soundfile",
    "pydot",
    "pydotplus",
    "graphviz",
    "python-dotenv",
    "requests",
    "mutagen",
    "pyfiglet",
    "pytorch-lightning",
    "test-tube",
    "torchsummary",
    "nnAudio",
    "addict",
    "numba==0.48.0",
    "cached-property",
    "more_itertools",
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
