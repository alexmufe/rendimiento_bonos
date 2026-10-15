# Mexico's yield curve: pattern, forecasts and the value of the pattern

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexmufe/rendimiento_bonos/blob/main/mexico_yield_curve.ipynb)

**Run it in Google Colab (nothing to install):** open the notebook with the badge above, or with https://colab.research.google.com/github/alexmufe/rendimiento_bonos/blob/main/mexico_yield_curve.ipynb, and choose *Runtime > Run all*. The first cell downloads this repository (data and code) and the notebook runs from start to finish in about a minute. It ends with a traceability table that compares each number in the brief with the value computed here.

**Run it locally:**

```bash
git clone https://github.com/alexmufe/rendimiento_bonos.git
cd rendimiento_bonos
pip install -r requirements.txt
jupyter notebook mexico_yield_curve.ipynb
```

`data/` holds the auction rates as delivered by Tukan; the `.py` files are the modules the notebook calls (data corrections, external and survey data with their sources, models and tests, figures). Tables and figures are written to `outputs/` when the notebook runs.

Code under the MIT License. The data belong to their sources.
