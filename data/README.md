# Data

All data in this project is **synthetic**. No real cases, customers or third-party datasets are included.

- `sample/expedientes_sample.csv`: first 1,000 rows of the generated input dataset (illustrative only).
- The full 60,000-row dataset and the model outputs are regenerated locally:

```bash
cd src
python genera_csv.py      # -> expedientes_gyar.csv
python motor_predict.py   # -> ips_modelo.csv, baseline_aleatorio.csv, *.png
```

`genera_csv.py` uses `np.random.seed(42)`, so the input is reproducible.

## Input columns
| Column | Description |
|---|---|
| `expediente_id` | Case identifier (`EXP_000000`, ...) |
| `dias_impago` | Days overdue (1-90) |
| `importe` | Amount (1,000-30,000), log-normal |
| `prioridad_base` | Base priority (1-5) |
| `historial` | Number of previous management attempts (0-5), correlated with days overdue |
| `reactivacion` | 1 if the case was reactivated, else 0 |
