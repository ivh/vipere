# vipere - Telluric correction for CRIRES+ spectra

Forked and adapted from [viper](https://github.com/mzechmeister/viper) by Zechmeister & Koehler.
Simplified to a single-file tool focused on telluric removal and stellar template creation for CRIRES+ data.

Author: Alexis Lavail (with help from Claude)

## Requirements

- Python >= 3.10
- astropy, matplotlib, numpy, scipy

## Installation

Install as a global command with [uv](https://docs.astral.sh/uv/):
```bash
uv tool install -e /path/to/vipere
```

The `-e` (editable) flag means changes to `vipere.py` take effect immediately without reinstalling.
You can then run `vipere` from anywhere on your system.

## Usage

```bash
vipere "data/WASP18/cr2res*.fits" \
  -createtpl -telluric add -tsig 10 -tpl_wave tell \
  -deg_norm 2 -deg_wave 2 -oset 1:28 -o data/WASP18/tpl1
```

Alternatively, run directly without installing:
```bash
uv run vipere.py "data/WASP18/cr2res*.fits" -createtpl -telluric add -oset 1:28 -o output
```

See `vipere -?` for all options.

### Key flags

| Flag | Description | Default |
|------|-------------|---------|
| `-o` | Output basename for result files | `tmp` |
| `-oset` | Order index range (e.g. `1:28` for all 27 orders) | `1:28` |
| `-nset` | Spectrum index range | `:` (all) |
| `-createtpl` | Create a stellar template from multiple observations | off |
| `-telluric` | Telluric mode: `add` (per-molecule coefficients) or `add2` (combined non-water coefficient) | off |
| `-global_atm` | Fit atmosphere globally across all orders (shared RV + atm coefficients) | off |
| `-tellshift` | Allow a wavelength shift of the telluric model | off |
| `-tell_bic` | BIC threshold for telluric model selection (0 to disable) | 10 |
| `-tsig` | Relative sigma for weighting telluric regions | 1 |
| `-tpl_wave` | Output wavelength grid: `initial`, `berv`, or `tell` | `initial` |
| `-tpl_noRV` | Do not apply stellar RV shift to telluric-corrected spectrum (with `-createtpl`) | off |
| `-deg_norm` | Polynomial degree for flux normalisation | 3 |
| `-deg_wave` | Polynomial degree for wavelength solution | 3 |
| `-deg_bkg` | Polynomial degree for background model | 0 |
| `-oversampling` | Oversampling factor for the template | auto |
| `-ip` | IP model: `g`, `ag`, `sg`, `bg`, `mg`, `mcg` | `g` |
| `-kapsig` | Kappa-sigma clipping values per stage (0 = no clipping) | 0 |
| `-kapsig_ctpl` | Kappa-sigma for outlier clipping in template creation | 0.6 |
| `-plot` | Plot level: 0 = off, 1 = with pause, 2 = without pause (saves PNG) | 0 |
| `-rv_guess` | Initial RV guess [km/s] | 1.0 |
| `-vcut` | Trim observation to range valid for model [km/s] | 100 |
| `-fix` | Fix parameters (e.g. `-fix wave` for stabilised instruments) | none |
| `-molec` | Molecules to include (`all` for automatic selection) | `all` |
| `-nexcl` | Ignore spectra matching string pattern | none |
| `-wgt` | Weighted fit: `error` (use data errors) or `tell` (upweight tellurics) | off |
| `-config_file` | YAML config file to override defaults | none |

## Citation

If you use this tool, please cite the original viper pipeline:
https://ui.adsabs.harvard.edu/abs/2021ascl.soft08006Z
