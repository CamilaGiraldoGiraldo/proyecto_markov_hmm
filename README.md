# HMM y cadenas de Markov para mantenimiento de redes comunitarias

Actividad didáctica 2-m2. Implementación desde cero (NumPy/SciPy) de un HMM discreto
(Forward, Backward, Viterbi, Baum-Welch) y de una cadena de Markov visible, con
simulación de 4 escenarios y análisis estadístico.

## Modelo
- Estados ocultos S = {Óptimo, Degradado, Falla}; observaciones V = {Normal, Lento, Intermitente, Sin respuesta}.
- Parámetros (A, B, π) en `run_experiments.py` (A_BASE, B_BASE, PI0, SCENARIOS).

## Archivos
- `hmm_markov.py`: clases `HMM` y `MarkovChain`, utilidades (distribución estacionaria, tiempos de alcance, alineación de estados).
- `run_experiments.py`: experimentos, tablas, figuras (300 dpi) y verificación de reproducibilidad.
- `results/`: CSV crudos, resumen estadístico, pruebas t, sensibilidad, figuras, entorno.

## Ejecución
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py            # completo: 100 réplicas por escenario (~4 min)
python run_experiments.py --reps 5   # prueba rápida
```

## Reproducibilidad
Todas las semillas derivan de `SEED_BASE = 20261106` vía `numpy.random.SeedSequence`.
Entorno usado: Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2.
SHA-256 de `results/resultados_crudos.csv`: `390982cc92ec2adefecfae5cd7f9ca12e953f798f29c75959160960ae89813e0`.

## Sugerencia de versionado
`git init && git add . && git commit -m "HMM mantenimiento redes"` y publicar en GitHub/Zenodo.
