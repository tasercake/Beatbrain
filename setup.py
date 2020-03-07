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
]

dev_requirements = [
    "black",
    "nb_black",
    "parametrized",
    "pytest",
    "pre-commit",
    "jupyter",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beatbrain",
    version="0.0.2",
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
