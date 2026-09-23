# OGInfoSP

This repository contains the Python implementation of OGInfoSP and the code to reproduce the experiments reported in the paper [“Selecting Informative Conformal Prediction Sets with an Optimized FCR-Controlled Approach”](https://arxiv.org/abs/2605.22004), by Israela Solomon, Etienne Roquain, Saharon Rosset, and Ruth Heller.

## Repository structure

- `simulation/`: algorithms, data generation, and experiment runner.
- `cifar/`: CIFAR-10 model code and trained models used by the experiments.
- `visualization/`: scripts used to generate the paper figures.
- `viewer/`: interactive viewer for the simulation results.

## Installation

```powershell
git clone https://github.com/israela/OGInfoSP.git
cd OGInfoSP
```

Alternatively, select **Code > Download ZIP** on GitHub, extract the downloaded ZIP, and open a terminal in the extracted repository folder.

Then install the required packages:

```powershell
python -m pip install -r requirements.txt
```

## Reproducing the results

The experiment configurations used for the paper are defined at the bottom of `simulation/main.py`.

From the repository root, run:

```powershell
python -m simulation.main
```

To generate the experiment figures, run:

```powershell
python -m visualization.plot_July_2026_paper_figures
```

Running the complete configuration can be computationally expensive, so the saved results are also provided in `simulation-results-July-2026.zip`.

## Interactive results viewer

We provide an interactive viewer for exploring and comparing the simulation results. See [viewer/README.md](viewer/README.md) for setup and launch instructions.

## License

This project is licensed under the [MIT License](LICENSE).
