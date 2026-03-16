# ML Framework for Constructing Heterogeneous Contact Networks

This repository contains the code accompanying the paper:

> **A Machine Learning Framework for Constructing Heterogeneous Contact Networks: Implications for Epidemic Modelling**

The framework uses Gaussian Mixture Models (GMMs) and Stochastic Block Models (SBMs) to construct realistic age and duration structured contact networks from survey data, and simulates epidemic dynamics on those networks using stochastic SEIR models.

---

## Requirements

- Python 3.8+
- Rust (for compiling the underlying simulation code)
- [Maturin](https://www.maturin.rs/) (for building the Python-Rust bindings)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Rust via [rustup](https://rustup.rs/) if not already installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install Maturin:

```bash
pip install maturin
```

---

## Compiling the Rust Code

The core network construction and epidemic simulation code is written in Rust and exposed to Python via `nd_python_avon`. To compile it, run the following from the root of the repository:

```bash
maturin develop --release
```

Do not worry about the warnings from running Maturin! They are stylistic rather than functional. <br>
This builds the Rust extension in release mode (optimised) and installs it into your current Python environment. You only need to do this once, or again after any changes to the Rust source.

---

## Input Data

The `input_data/` directory contains preprocessed contact survey datasets required to run the examples:

- `input_data/egos/` — ego network representations of survey respondents
- `input_data/gmm/` — pre-fitted optimal GMM component counts (selected via BIC)
- `input_data/durations/` — contact duration distributions for short contacts

Four datasets are provided, corresponding to different survey waves:

| Key | Description |
|---|---|
| `comix3` | CoMix – Reopen wave |
| `comixa` | CoMix – Lockdown 2020 |
| `comixb` | CoMix – Lockdown 2021 |
| `poly` | POLYMOD pre-pandemic data |

---

## Example Usage

A Jupyter notebook is provided demonstrating the full pipeline in "example.ipynb". The key steps are outlined below.

### 1. Load a Dataset

```python
import json
import numpy as np
import nd_python_avon as nd
import matplotlib.pyplot as plt

datas = ['comix3', 'comixa', 'comixb', 'poly']
data = datas[0]  # choose a dataset
buckets = np.array([5, 12, 18, 30, 40, 50, 60, 70])  # age brackets

# Load pre-fitted optimal GMM components (selected via BIC) - log scaled gmm and 0-1 hour contacts squashed to reduce directional inaccuracies in GMM fitting
with open(f'input_data/gmm/optimal_components_{data}_log_smalldur.json', 'r') as f: 
    tmp = json.load(f)
optimal_num_components = tmp[data]

# Load ego networks and duration properties to expand 0-1hour in simulation
with open(f'input_data/egos/{data}_dur_small.json', 'r') as f:
    egos = json.load(f)
props = np.genfromtxt(f'input_data/durations/{data}.csv', delimiter=',')
```

You can also fit to your **own dataset** using:

```python
import pandas as pd
egos, contact_matrix = nd.fit_to_data(
    input_file_path='input_data/your_data.csv',
    buckets=np.array([5, 12, 18, 30, 40, 50, 60, 70]),
    duration=True
)
```

Your CSV should contain one row per contact, with columns for respondent ID, respondent age, contact ID, contact age ("cnt_age_exact" or "cnt_age_min"/"cnt_age_max"), and duration. In line with the syntax presented in [socialcontactdata.org](socialcontactdata.org).

### 2. Sample a Synthetic Population

```python
n = 1000  # network size

# Age group population fractions from UK census data
partitions = [0.058*n, 0.145*n, 0.212*n, 0.364*n, 0.497*n,
              0.623*n, 0.759*n, 0.866*n, n]

# Sample ego networks for the synthetic population using the GMM
samples = nd.sample_egos_gmm(
    egos=egos,
    partitions=partitions,
    optimal_num_components=optimal_num_components
)
```

### 3. Run a Stochastic SEIR Epidemic

```python
tau = 1        # transmission rate
gamma = 1/4    # recovery rate
sigma = 1      # latency rate
num_infected = 1

output = nd.small_dur_gillesp(
    samples,
    partitions=partitions,
    num_dur=3,
    tau=tau,
    gamma=gamma,
    sigma=sigma,
    num_infec=num_infected,
    props=props.tolist()
)
```

The output dictionary contains details of the network created (e.g., adjacency matrix, frequency distribution and individual age groups), and the outbreak outputs (SEIR events with time points, generation infected, number of secondary cases and between who each transmission occured)


### 4. Visualise Epidemic Trajectories

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for i in range(100):
    res = nd.small_dur_gillesp(
        samples, partitions=partitions, num_dur=3,
        tau=tau, gamma=gamma, sigma=sigma,
        num_infec=num_infected, props=props.tolist()
    )
    ax.plot(res["ts"], [a[2]/n for a in res['sir']],
            color='tab:blue', alpha=0.4, linewidth=1)

ax.set_xlabel("Time (days)")
ax.set_ylabel("Proportion Infected")
ax.set_ylim(0, ax.get_ylim()[1])
ax.set_xlim(0, ax.get_xlim()[1])
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
plt.show()
```

---

## Repository Structure

```
├── duration+ages/          # Simulation results
├── experiments/            # Files ran on HPC cluster to collect simulation results
├── input_data/
│   ├── egos/               # Preprocessed ego networks per dataset
│   ├── gmm/                # Optimal GMM component counts
│   └── durations/          
├── output_data/            # figure container and EMD error
├── src/                    # Rust source code for network construction and simulation
├── Cargo.toml              # Rust dependencies
├── Example_Notebook.ipynb  
├── Figures.ipynb
├── nd_python_avon.py       # Python interface to the Rust library
└── requirements.txt        # Python dependencies
```

---

## Citation

If you use this code in your work, please cite:

```

```

---

## License

This project is licensed under CC BY 4.0. See [LICENSE](LICENSE) for details.
