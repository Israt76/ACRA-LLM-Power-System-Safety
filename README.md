# ACRA: Coordinated Semantic–Telemetry Attacks on DC/LODF Safety Shields for LLM-Based Power-System Agents

Reproducibility repository for the ACRA manuscript by **Israt Jahan Aunika** and **Stefan Robila**, Montclair State University.

ACRA studies a coordinated semantic–telemetry attack against DC/LODF safety shields used by LLM-based power-system agents. The repository contains the manuscript source, experiment code, frozen result files, notebooks, figures, and verification records needed to reproduce the reported results.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── requirements-llm.txt
├── manuscript/
│   └── ACRA_access.tex
├── notebooks/
│   ├── ACRA_Complete_Notebook.ipynb
│   ├── physics_reproduce.ipynb
│   ├── llm_reproduce.ipynb
│   └── llm_clean_rerun_colab.ipynb
├── code/
│   ├── acra_physics.py
│   ├── check_results.py
│   ├── defgreedy.py
│   ├── exp_magnitude_minlinf.py
│   ├── stats_permutation.py
│   ├── exp_bdd_noise.py
│   ├── exp_scaleup2.py
│   └── ...
├── data/
│   ├── ieee14_budget.csv
│   ├── ieee14_magnitude_per_target.csv
│   ├── llm_four_model_raw.jsonl
│   └── ...
├── figures/
│   └── final manuscript figures
└── docs/
    └── VERIFICATION_LOG.txt
```

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

For the four-model LLM rerun:

```bash
pip install -r requirements-llm.txt
```

## Core reproduction checks

Run commands from the repository root:

```bash
python code/check_results.py
python code/defgreedy.py
python code/exp_magnitude_minlinf.py
python code/stats_permutation.py
python code/exp_bdd_noise.py
```

The frozen verification record is provided in [`docs/VERIFICATION_LOG.txt`](docs/VERIFICATION_LOG.txt).

## Headline result map

| Manuscript result | Reproduction source |
|---|---|
| IEEE-14 minimum compromised-meter budgets | `code/check_results.py`, `data/ieee14_budget.csv` |
| Minimum per-meter injection magnitude | `code/exp_magnitude_minlinf.py`, `data/ieee14_magnitude_per_target.csv` |
| LLM behavioral results | `data/llm_four_model_raw.jsonl`, `data/llm_four_model_summary.csv` |
| IEEE-39 / IEEE-118 scale-up | `code/exp_scaleup2.py`, `data/exp_scaleup2.json` |
| Greedy secure-meter defense | `code/defgreedy.py`, `data/ieee14_defense_greedy.csv` |
| Noisy bad-data-detection experiment | `code/exp_bdd_noise.py`, `data/ieee14_bdd_noise.csv` |
| Cluster-level permutation analysis | `code/stats_permutation.py` |
| Defense robustness band | `code/fig_defense_band.py`, `data/fig_defense_band.json` |

## Notes

- The DC/LODF attack results are scoped to the approximation-based verifier studied in the manuscript.
- The repository includes frozen outputs so reported tables and figures can be checked independently of solver variability.
- The balanced four-model LLM rerun is the only workflow intended for a GPU environment.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update the journal, volume, pages, and DOI after publication.
