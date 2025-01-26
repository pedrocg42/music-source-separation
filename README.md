<h1 align='center'>Continual Learning for (Line-Level) Optical Music Recognition</h1>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.11.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white'alt='PyTorch'>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#installation'>Installation</a> •
  <a href='#experiments'>Experiments</a>
</p>


## About

This repository serves as a template for research experiments repositories

## Installation

### Set Up with uv

Follow these steps to set up the environment using [uv](https://docs.astral.sh/uv/):

```bash

# Create and activate a new uv environment
curl -LsSf https://astral.sh/uv/install.sh | sh # (linux or mac)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

uv venv --python 3.11
source .venv/bin/activate # (linux or mac)
.venv/Scripts/activate # Windows

# Install the required packages
uv pip install -e . --no-cache-dir
```