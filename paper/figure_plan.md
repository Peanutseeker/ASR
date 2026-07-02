# Figure Plan

## Main Paper Figures

1. **Three-panel robustness figure**  
   Purpose: show the central robustness result under clean, 20 dB, 10 dB, and 5 dB conditions, then explain the 5 dB failure mode.  
   LaTeX use: one two-column figure with WER, CER, and edit-breakdown panels.  
   Generation command:
   ```bash
   uv run python -B scripts/17_make_paper_figures.py \
     --metrics-long results/tables/metrics_long.csv \
     --figures-dir paper/figures
   ```
   Assets:
   - `figures/paper_robustness_main.pdf`
   - `figures/paper_robustness_main.svg`
   - `figures/paper_robustness_main.png`

## Supporting Figure

2. **Noise penalty figure**  
   Purpose: useful for presentation slides or an appendix; not used in the main paper because the three-panel figure already carries the central evidence.  
   Asset:
   - `figures/paper_noise_penalty.pdf`
   - `figures/paper_noise_penalty.svg`
   - `figures/paper_noise_penalty.png`

## Table

1. **WER summary table**  
   Purpose: give exact values for the four systems and four acoustic conditions.  
   Source:
   - `../results/tables/metrics_summary.csv`
