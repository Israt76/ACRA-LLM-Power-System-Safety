# ACRA — Reproducibility Package

Supplementary code, data, and logs for *ACRA: Coordinated Semantic–Telemetry
Attacks on DC/LODF Safety Shields for LLM Grid Agents* (IEEE Access submission).
Every headline number in the paper regenerates from the scripts below.

## Layout
- `code/` — grid-agnostic physics core (`acra_physics.py`) and all experiment scripts.
- `data/` — frozen result files (CSV + JSON) and the raw LLM trial log.
- `figures/` — all six paper figures.
- `check_results.py` — recomputes IEEE-14 budgets and checks them against the CSV.
- `physics_reproduce.ipynb`, `llm_reproduce.ipynb` — notebook drivers.
- `llm_clean_rerun_colab.ipynb` — GPU driver for the balanced four-model LLM run.

## Requirements
`python>=3.10`, `numpy`, `scipy>=1.11` (MILP), `pandapower`. The LLM rerun needs a
GPU, `transformers`, `bitsandbytes` (4-bit); it is the only GPU-dependent step.

## One-command checks (CPU, minutes)
Run ALL commands from this directory (the package root), not from `code/`:
```
python check_results.py                      # Table 1 budgets  -> ALL MATCH
python code/defgreedy.py                     # Table 6 defense -> min_set [24, 12, 11, 0]
python code/exp_magnitude_minlinf.py         # Table 2 reproducible rho* magnitudes
python code/stats_permutation.py             # Sec VI  -> gap 0.543, permutation p < 1e-4
```
A full end-to-end run of these checks is recorded in `VERIFICATION_LOG.txt`.

## Headline-number map (paper -> generator)
| Paper claim | Value | Script / data |
|---|---|---|
| Table 1, IEEE-14 budgets | k_app 4–8, k_tel 11–18 / infeasible | `check_results.py`, `data/ieee14_budget.csv` |
| Table 2, injection magnitude | reproducible rho* 0.12–0.84 (k_app from Table 1) | `exp_magnitude_minlinf.py`, `data/ieee14_magnitude_per_target.csv` |
| Table 3, LLM success | JOINT only; 0.25–1.00 | `data/llm_four_model_raw.jsonl`, `llm_four_model_summary.csv` |
| Tables 4–5, IEEE-39 / IEEE-118 scale-up | 6/10 and 3/6 telemetry-only infeasible, 0 timeouts | `exp_scaleup2.py`, `data/exp_scaleup2.json` |
| Table 6, greedy defense | meters 24,12,11,0 → 0 attackable | `defgreedy.py`, `data/ieee14_defense_greedy.csv` |
| Fig. 2(b), naive band | mean + min–max over 30 seeds | `fig_defense_band.py`, `data/fig_defense_band.json` |
| Noisy BDD (Sec VI) | FA 1.07%; random detected 92–100% | `exp_bdd_noise.py`, `data/ieee14_bdd_noise.csv` |
| LLM separation (Sec VI) | gap 0.543, permutation p < 1e-4 | `stats_permutation.py` |

## Notes on consistency (addressing reviewer items)
- **Table 2 magnitude is reproducible.** The minimum-meter MILP fixes the meter
  *count* (`k_app`, Table 1), not the injection magnitude, so the magnitude of a
  specific k_app-meter attack is solver-dependent. Table 2 therefore reports `rho*`,
  the minimum achievable max per-meter relative magnitude (an LP, `exp_magnitude_minlinf.py`),
  which is deterministic. k_app is shown alongside as the Table 1 budget.
  (`exp_magnitude_recompute.py` is retained as a superseded min-meter variant.)
- **Defense method.** Table 6 uses the marginal greedy set-cover of Algorithm 1
  (`defgreedy.py`), which re-solves the attack each step. The earlier static
  frequency ranking (`exp_defense_rank.py`) is retained only as a superseded baseline.
  The four-meter set is *not* claimed to be of minimum cardinality.
- **Frozen scale-up.** IEEE-39/118 numbers are frozen in `data/exp_scaleup2.json`
  (60 s solver limit; infeasibility proven, 0 timeouts) for run-to-run consistency.

## LLM raw log schema (`data/llm_four_model_raw.jsonl`, 6406 records)
`model, model_id, config_hash, target, condition, doc (wording), sample, bdd_flag,
action, induced, verifier_allow, truly_unsafe, attack_success, raw`.
