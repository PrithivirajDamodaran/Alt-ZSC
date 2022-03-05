import setuptools
import os

os.system("pip install git+https://github.com/openai/CLIP.git") 
os.system("pip install sentence-transformers")


setuptools.setup(
    name="Alt-ZSC",
    version="1.0",
    author="PithiviDa @ Donkey Stereotype",
    author_email="",
    description="Zeroshot Text Classification",
    long_description="Alternate implementation for Zero Shot Text Classification",
    url="https://github.com/PrithivirajDamodaran/Alt-ZSC.git",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
