# Robust Bond Portfolio Construction via Convex-Concave Saddle Point Optimization
[![Linters](https://github.com/cvxgrp/robust_bond_portfolio/actions/workflows/linter.yml/badge.svg)](https://github.com/cvxgrp/robust_bond_portfolio/actions/workflows/linter.yml)

This repo accompanies our [paper](https://arxiv.org/abs/2212.02570) and provides code and data to replicate all results.

## Getting started
Clone the repo and run
```bash
pip install -e .
```
from the root folder.

## Running the code
To recreate the examples, simply run
```bash
python run_examples.py
```
which will show all figures and print the numerical results to the terminal.

> **Note**
> The examples require the Mosek solver (academic licenses are available).

## Code snippets
All code snippets presented in the paper are maintained as test cases is `tests/test_snippets.py` and can be copied from there.

## Citing
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:

```BibTeX
@misc{luxenberg2022robustbond,
  author = {Luxenberg, Eric and Schiele, Philipp and Boyd, Stephen},
  doi = {10.48550/ARXIV.2212.02570},
  title = {Robust Bond Portfolio Construction via Convex-Concave Saddle Point Optimization},
  url = {https://arxiv.org/abs/2212.02570},
  year = {2022}
}
```
