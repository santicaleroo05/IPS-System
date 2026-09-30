# How to run

Step-by-step guide to run the IPS System locally. Requires Python 3.9+ (tested with Python 3 and the latest versions of the libraries in `requirements.txt`).

## 1. Install

```bash
git clone git@github.com:santicaleroo05/IPS-System.git
cd IPS-System

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the pipeline

The scripts read and write files in the **current directory**, so always run them from `src/`, in this order.

```bash
cd src
```

**Step 1: generate the synthetic dataset**

```bash
python genera_csv.py
```
Creates `expedientes_gyar.csv` (60,000 cases, seed 42, so it is reproducible).

**Step 2: run the engine**

```bash
python motor_predict.py
```
Trains the Random Forest, scores and ranks the cases, assigns them to the 39 managers, runs the random baseline, prints the KPIs and creates:

- `ips_modelo.csv`
- `baseline_aleatorio.csv`
- `comparacion_ips_vs_random.png`, `kpi_desviacion.png`, `kpi_productividad_por_gestor.png`

The script calls `plt.show()`, so it opens chart windows: close each one to let it continue. To skip the windows (e.g. on a server):

```bash
MPLBACKEND=Agg python motor_predict.py      # Windows (PowerShell): $env:MPLBACKEND="Agg"; python motor_predict.py
```

Expected console output (seed 42):

```
Desviación de carga: 46.61
% intervención temprana: 65.68%
Productividad media expedientes/gestor: 307.69
Índice de Afinidad (Casos Críticos -> Expertos): 49.85%
Desviación carga (Random): 103.38
Mejora en equilibrio de carga: 54.92%
```

**Step 3: launch the dashboard**

```bash
streamlit run app_predict.py
```
Opens at http://localhost:8501. It needs `ips_modelo.csv` and `baseline_aleatorio.csv` from Step 2 in the same folder.

## Earlier versions (optional)

Scripts in `iterations/` are earlier, rule-based versions. They also read `expedientes_gyar.csv` from the current directory, so copy it there first:

```bash
cd iterations
cp ../src/expedientes_gyar.csv .
python motor_total.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: expedientes_gyar.csv` | Run `genera_csv.py` first, and run every script from `src/`. |
| `FileNotFoundError: ips_modelo.csv` in the dashboard | Run `motor_predict.py` first. |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Script seems stuck after printing KPIs | Close the Matplotlib windows, or use `MPLBACKEND=Agg`. |
