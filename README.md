# vipere - Telluric correction for CRIRES+ spectra

Forked and adapted from [viper](https://github.com/mzechmeister/viper) by Zechmeister & Koehler.
Simplified to a single-file tool focused on telluric removal and stellar template creation for CRIRES+ data.

Author: Alexis Lavail (with help from Claude)

## Requirements

- Python >= 3.10
- astropy, matplotlib, numpy, scipy

## Usage

Run directly with [uv](https://docs.astral.sh/uv/):
```bash
uv run vipere.py "data/WASP18/cr2res*.fits" \
  -createtpl -telluric add -tsig 10 -tpl_wave tell \
  -deg_norm 2 -deg_wave 2 -oset 1:28 -o data/WASP18/tpl1
```

Or install and run:
```bash
uv pip install -e .
vipere "data/WASP18/cr2res*.fits" -createtpl -telluric add -oset 1:28 -o output
```

See `vipere -?` for all options.

## Citation

If you use this tool, please cite the original viper pipeline:
https://ui.adsabs.harvard.edu/abs/2021ascl.soft08006Z
