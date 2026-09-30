# Earlier iterations

Scripts kept as-is to show how the model evolved. They are **not** the final version
(see `../src/`). They expect `expedientes_gyar.csv` (and, for the apps, the generated
`ips_modelo.csv` / `baseline_aleatorio.csv`) in the directory they are run from.

| File | Description |
|---|---|
| `motor.py` | First version of the rule-based scoring engine. |
| `motor_comparado.py`, `motor_final.py` | Rule-based engine (weights 0.40 days / 0.25 priority / 0.20 amount / 0.15 history) with a random-assignment baseline. |
| `motor_total.py` | Rule-based engine with reactivation boost, manager assignment, KPIs and plots, **without** the ML risk score. |
| `app.py` | First Streamlit dashboard (rule-based version). |
