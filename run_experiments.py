"""
run_experiments.py
==================
Experimentos del artículo: salud de enlaces de internet comunitario modelada con
una cadena de Markov visible y con un HMM.

Estados ocultos  S = {Óptimo, Degradado, Falla}
Observaciones    V = {Normal, Lento, Intermitente, Sin respuesta}  (sondeos tipo ping)

Uso:
    python run_experiments.py            # experimento completo (100 réplicas)
    python run_experiments.py --reps 5   # prueba rápida

Todas las semillas derivan de SEED_BASE mediante np.random.SeedSequence, por lo que
la ejecución es completamente reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from hmm_markov import (
    HMM,
    MarkovChain,
    align_states,
    mean_hitting_time,
    stationary_distribution,
)

SEED_BASE = 20261106
STATES = ["Óptimo", "Degradado", "Falla"]
SYMBOLS = ["Normal", "Lento", "Intermitente", "Sin respuesta"]
N, M = 3, 4
T_DEFAULT = 500
PI0 = np.array([0.80, 0.15, 0.05])

B_BASE = np.array(
    [
        [0.80, 0.15, 0.04, 0.01],
        [0.20, 0.45, 0.30, 0.05],
        [0.02, 0.08, 0.30, 0.60],
    ]
)
A_BASE = np.array(
    [
        [0.95, 0.04, 0.01],
        [0.10, 0.80, 0.10],
        [0.30, 0.10, 0.60],
    ]
)

SCENARIOS = {
    "S1 Base": (A_BASE, B_BASE),
    "S2 Persistente y separable": (
        np.array([[0.98, 0.015, 0.005], [0.05, 0.90, 0.05], [0.15, 0.05, 0.80]]),
        np.array(
            [
                [0.90, 0.08, 0.015, 0.005],
                [0.05, 0.70, 0.22, 0.03],
                [0.01, 0.04, 0.15, 0.80],
            ]
        ),
    ),
    "S3 Emisiones solapadas": (
        A_BASE,
        np.array(
            [
                [0.55, 0.25, 0.15, 0.05],
                [0.30, 0.35, 0.25, 0.10],
                [0.15, 0.25, 0.30, 0.30],
            ]
        ),
    ),
    "S4 Red inestable": (
        np.array([[0.85, 0.10, 0.05], [0.15, 0.65, 0.20], [0.25, 0.15, 0.60]]),
        B_BASE,
    ),
}

# Paleta apta para daltonismo (Okabe-Ito)
C_BLUE, C_ORANGE, C_GREEN, C_GREY, C_RED = "#0072B2", "#E69F00", "#009E73", "#7F7F7F", "#D55E00"

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.frameon": False,
    }
)


# --------------------------------------------------------------------------- #
# Utilidades estadísticas
# --------------------------------------------------------------------------- #
def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, m
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m - h, m + h


def summarize(df, group_cols, metric_cols):
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metric_cols:
            lo, hi = ci95(g[m])
            rows.append(
                dict(
                    zip(group_cols, key),
                    metrica=m,
                    n=len(g),
                    media=g[m].mean(),
                    desv_est=g[m].std(ddof=1),
                    minimo=g[m].min(),
                    maximo=g[m].max(),
                    ic95_inf=lo,
                    ic95_sup=hi,
                )
            )
    return pd.DataFrame(rows)


def rng_for(*ids):
    return np.random.default_rng(np.random.SeedSequence([SEED_BASE, *ids]))


def permute(model: HMM, perm):
    return HMM(model.A[np.ix_(perm, perm)], model.B[perm], model.pi[perm])


# --------------------------------------------------------------------------- #
# Una réplica de simulación
# --------------------------------------------------------------------------- #
def run_rep(s_idx, rep, A, B, T=T_DEFAULT, restarts=2, n_iter=80):
    true = HMM(A, B, PI0)
    rng = rng_for(s_idx, rep, 0)
    _, o_tr = true.sample(T, rng)
    q_te, o_te = true.sample(T, rng)

    # --- Modelos ajustados con la secuencia de entrenamiento ---
    learned, hist, n_it = HMM.baum_welch(
        [o_tr], N, M, rng_for(s_idx, rep, 1), n_iter=n_iter, tol=1e-5, restarts=restarts
    )
    perm = align_states(true.B, learned.B)
    learned_al = permute(learned, perm)
    mc = MarkovChain.fit(o_tr, M)
    freq = (np.bincount(o_tr, minlength=M) + 1.0)
    freq /= freq.sum()

    # --- Decodificación de estados ocultos (solo HMM puede hacerlo) ---
    acc_vit_true = np.mean(true.viterbi(o_te) == q_te)
    acc_vit_learn = np.mean(learned_al.viterbi(o_te) == q_te)
    pi_inf = stationary_distribution(A)
    memoryless = np.argmax(pi_inf[:, None] * B, axis=0)[o_te]  # regla de Bayes sin memoria
    acc_memless = np.mean(memoryless == q_te)

    return dict(
        escenario=list(SCENARIOS)[s_idx],
        rep=rep,
        acc_viterbi_verdadero=acc_vit_true,
        acc_viterbi_aprendido=acc_vit_learn,
        acc_sin_memoria=acc_memless,
        ll_hmm_verdadero=true.loglik(o_te) / T,
        ll_hmm_aprendido=learned_al.loglik(o_te) / T,
        ll_markov_visible=mc.loglik(o_te) / T,
        ll_iid=float(np.log(freq[o_te]).sum()) / T,
        error_A_frobenius=float(np.linalg.norm(learned_al.A - A)),
        error_B_frobenius=float(np.linalg.norm(learned_al.B - B)),
        iter_baum_welch=n_it,
    )


# --------------------------------------------------------------------------- #
# Análisis teórico por escenario
# --------------------------------------------------------------------------- #
def theory_table():
    rows = []
    for name, (A, B) in SCENARIOS.items():
        pinf = stationary_distribution(A)
        h = mean_hitting_time(A, target=2)
        rows.append(
            {
                "escenario": name,
                "pi_inf_Optimo": pinf[0],
                "pi_inf_Degradado": pinf[1],
                "pi_inf_Falla": pinf[2],
                "disponibilidad_1_menos_pi_Falla": 1 - pinf[2],
                "permanencia_media_Optimo_pasos": 1 / (1 - A[0, 0]),
                "permanencia_media_Falla_pasos": 1 / (1 - A[2, 2]),
                "tiempo_medio_hasta_Falla_desde_Optimo": h[0],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensibilidad
# --------------------------------------------------------------------------- #
def sensitivity_params(reps, T=T_DEFAULT):
    """Decodificar con parámetros perturbados (mala especificación de A o de B)."""
    eps_grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    A, B = SCENARIOS["S1 Base"]
    true = HMM(A, B, PI0)
    rows = []
    for rep in range(reps):
        q, o = true.sample(T, rng_for(100, rep, 0))
        rng = rng_for(100, rep, 1)
        RA = rng.dirichlet(np.ones(N), size=N)
        RB = rng.dirichlet(np.ones(M), size=N)
        for eps in eps_grid:
            for which in ("A", "B"):
                if which == "A":
                    mod = HMM((1 - eps) * A + eps * RA, B, PI0)
                else:
                    mod = HMM(A, (1 - eps) * B + eps * RB, PI0)
                rows.append(
                    dict(
                        parametro=which,
                        epsilon=eps,
                        rep=rep,
                        acc_viterbi=np.mean(mod.viterbi(o) == q),
                        ll_por_simbolo=mod.loglik(o) / T,
                    )
                )
    return pd.DataFrame(rows)


def sensitivity_length(reps):
    """Efecto de la longitud de la secuencia de entrenamiento en Baum-Welch."""
    A, B = SCENARIOS["S1 Base"]
    true = HMM(A, B, PI0)
    rows = []
    for T in [50, 100, 200, 500, 1000, 2000]:
        for rep in range(reps):
            rng = rng_for(200, T, rep, 0)
            _, o_tr = true.sample(T, rng)
            q_te, o_te = true.sample(500, rng)
            learned, _, n_it = HMM.baum_welch(
                [o_tr], N, M, rng_for(200, T, rep, 1), n_iter=80, restarts=2
            )
            al = permute(learned, align_states(B, learned.B))
            rows.append(
                dict(
                    T_entrenamiento=T,
                    rep=rep,
                    error_A_frobenius=float(np.linalg.norm(al.A - A)),
                    acc_viterbi_aprendido=np.mean(al.viterbi(o_te) == q_te),
                    ll_test=al.loglik(o_te) / 500,
                )
            )
    return pd.DataFrame(rows)


def sensitivity_init(n_inits=50):
    """Mismo conjunto de datos, 50 inicializaciones aleatorias distintas de EM."""
    A, B = SCENARIOS["S1 Base"]
    true = HMM(A, B, PI0)
    rng = rng_for(300, 0, 0)
    _, o_tr = true.sample(T_DEFAULT, rng)
    q_te, o_te = true.sample(T_DEFAULT, rng)
    rows = []
    for k in range(n_inits):
        learned, _, n_it = HMM.baum_welch(
            [o_tr], N, M, rng_for(300, 1, k), n_iter=150, tol=1e-6, restarts=1
        )
        al = permute(learned, align_states(B, learned.B))
        rows.append(
            dict(
                init=k,
                ll_train=learned.loglik(o_tr) / T_DEFAULT,
                acc_viterbi=np.mean(al.viterbi(o_te) == q_te),
                error_A_frobenius=float(np.linalg.norm(al.A - A)),
                iteraciones=n_it,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def style_box(bp, colors):
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("black")


def fig_accuracy(df, path):
    names = list(SCENARIOS)
    meths = [
        ("acc_sin_memoria", "Clasificador sin memoria", C_GREY),
        ("acc_viterbi_aprendido", "HMM aprendido (Viterbi)", C_ORANGE),
        ("acc_viterbi_verdadero", "HMM verdadero (Viterbi)", C_BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.25
    for j, (col, lab, c) in enumerate(meths):
        data = [df.loc[df.escenario == n, col].values for n in names]
        pos = np.arange(len(names)) + (j - 1) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False)
        style_box(bp, [c] * len(names))
        ax.plot([], [], "s", color=c, label=lab)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names])
    ax.set_ylabel("Exactitud de decodificación de estados")
    ax.set_xlabel("Escenario")
    ax.set_title("Fig. 1. Exactitud de la decodificación del estado oculto (100 réplicas)")
    ax.legend(loc="lower left", ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_loglik(df, path):
    names = list(SCENARIOS)
    meths = [
        ("ll_iid", "Categórica i.i.d.", C_GREY),
        ("ll_markov_visible", "Cadena de Markov visible", C_GREEN),
        ("ll_hmm_aprendido", "HMM aprendido", C_ORANGE),
        ("ll_hmm_verdadero", "HMM verdadero", C_BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.2
    for j, (col, lab, c) in enumerate(meths):
        means, err = [], []
        for n in names:
            x = df.loc[df.escenario == n, col].values
            lo, hi = ci95(x)
            means.append(x.mean())
            err.append((hi - lo) / 2)
        ax.bar(np.arange(len(names)) + (j - 1.5) * w, means, w, yerr=err, capsize=2, color=c, label=lab)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names])
    ax.set_ylabel("Log-verosimilitud por símbolo (prueba)")
    ax.set_xlabel("Escenario")
    ax.set_title("Fig. 2. Capacidad predictiva fuera de muestra (media e IC 95 %)")
    lo = df[[m[0] for m in meths]].min().min()
    ax.set_ylim(lo - 0.08, df[[m[0] for m in meths]].max().max() + 0.05)
    ax.legend(loc="lower center", ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sens_params(sp, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (which, c, lab) in zip(axes, [("A", C_BLUE, "Perturbación de A"), ("B", C_ORANGE, "Perturbación de B")]):
        g = sp[sp.parametro == which]
        eps = sorted(g.epsilon.unique())
        mean = [g[g.epsilon == e].acc_viterbi.mean() for e in eps]
        lo = [ci95(g[g.epsilon == e].acc_viterbi)[0] for e in eps]
        hi = [ci95(g[g.epsilon == e].acc_viterbi)[1] for e in eps]
        ax.plot(eps, mean, "-o", color=c, ms=3)
        ax.fill_between(eps, lo, hi, color=c, alpha=0.25)
        ax.set_xlabel(r"Intensidad de perturbación $\varepsilon$")
        ax.set_title(lab)
    axes[0].set_ylabel("Exactitud de Viterbi")
    fig.suptitle("Fig. 3. Sensibilidad de la decodificación a errores en A y B (media e IC 95 %)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sens_length(sl, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    Ts = sorted(sl.T_entrenamiento.unique())
    for ax, col, c, yl in [
        (axes[0], "error_A_frobenius", C_RED, r"Error $\|\hat{A}-A\|_F$"),
        (axes[1], "acc_viterbi_aprendido", C_BLUE, "Exactitud de Viterbi (prueba)"),
    ]:
        mean = [sl[sl.T_entrenamiento == t][col].mean() for t in Ts]
        lo = [ci95(sl[sl.T_entrenamiento == t][col])[0] for t in Ts]
        hi = [ci95(sl[sl.T_entrenamiento == t][col])[1] for t in Ts]
        ax.plot(Ts, mean, "-o", color=c, ms=3)
        ax.fill_between(Ts, lo, hi, color=c, alpha=0.25)
        ax.set_xscale("log")
        ax.set_xlabel("Longitud de la secuencia de entrenamiento T")
        ax.set_ylabel(yl)
    fig.suptitle("Fig. 4. Efecto de la longitud de entrenamiento en Baum-Welch (media e IC 95 %)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_example(path):
    A, B = SCENARIOS["S1 Base"]
    true = HMM(A, B, PI0)
    q, o = true.sample(120, rng_for(400, 0, 0))
    path_v = true.viterbi(o)
    t = np.arange(len(q))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    axes[0].step(t, o, where="mid", color=C_GREY, lw=1)
    axes[0].set_yticks(range(M))
    axes[0].set_yticklabels(SYMBOLS, fontsize=7)
    axes[0].set_ylabel("Observación")
    axes[1].step(t, q + 0.04, where="mid", color=C_BLUE, lw=1.6, label="Estado real")
    axes[1].step(t, path_v - 0.04, where="mid", color=C_ORANGE, lw=1.2, ls="--", label="Estado decodificado (Viterbi)")
    axes[1].set_yticks(range(N))
    axes[1].set_yticklabels(STATES)
    axes[1].set_xlabel("Tiempo (sondeo)")
    axes[1].set_ylabel("Estado oculto")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[0].set_title("Fig. 5. Ejemplo (escenario S1): observaciones y recuperación del estado oculto")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100, help="réplicas por escenario")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figuras"), exist_ok=True)
    t0 = time.time()

    # 1) Escenarios
    rows = []
    for s_idx, (name, (A, B)) in enumerate(SCENARIOS.items()):
        for rep in range(args.reps):
            rows.append(run_rep(s_idx, rep, A, B))
        print(f"[{time.time() - t0:6.1f}s] {name} listo", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out, "resultados_crudos.csv"), index=False)

    metric_cols = [c for c in raw.columns if c not in ("escenario", "rep")]
    summ = summarize(raw, ["escenario"], metric_cols)
    summ.to_csv(os.path.join(out, "resumen_estadistico.csv"), index=False)

    theory = theory_table()
    theory.to_csv(os.path.join(out, "analisis_teorico.csv"), index=False)

    # Comparaciones entre escenarios / modelos (pruebas pareadas)
    tests = []
    for name in SCENARIOS:
        g = raw[raw.escenario == name]
        for a, b, lab in [
            ("ll_hmm_aprendido", "ll_markov_visible", "LL: HMM aprendido vs Markov visible"),
            ("acc_viterbi_aprendido", "acc_sin_memoria", "Exactitud: Viterbi aprendido vs sin memoria"),
        ]:
            t_stat, p = stats.ttest_rel(g[a], g[b])
            tests.append(
                dict(escenario=name, comparacion=lab, dif_media=(g[a] - g[b]).mean(), t=t_stat, p_valor=p)
            )
    pd.DataFrame(tests).to_csv(os.path.join(out, "pruebas_t_pareadas.csv"), index=False)

    # 2) Sensibilidad
    sp_reps = max(5, args.reps // 2)
    sp = sensitivity_params(sp_reps)
    sp.to_csv(os.path.join(out, "sensibilidad_parametros.csv"), index=False)
    print(f"[{time.time() - t0:6.1f}s] sensibilidad A/B lista", flush=True)
    sl = sensitivity_length(max(5, args.reps // 3))
    sl.to_csv(os.path.join(out, "sensibilidad_longitud.csv"), index=False)
    print(f"[{time.time() - t0:6.1f}s] sensibilidad T lista", flush=True)
    si = sensitivity_init(50 if args.reps >= 50 else 10)
    si.to_csv(os.path.join(out, "sensibilidad_inicializacion.csv"), index=False)
    print(f"[{time.time() - t0:6.1f}s] sensibilidad init lista", flush=True)

    # 3) Figuras (300 dpi)
    fd = os.path.join(out, "figuras")
    fig_accuracy(raw, os.path.join(fd, "fig1_exactitud.png"))
    fig_loglik(raw, os.path.join(fd, "fig2_loglik.png"))
    fig_sens_params(sp, os.path.join(fd, "fig3_sensibilidad_AB.png"))
    fig_sens_length(sl, os.path.join(fd, "fig4_longitud.png"))
    fig_example(os.path.join(fd, "fig5_ejemplo.png"))

    # 4) Reproducibilidad: repetir las primeras réplicas y comparar bit a bit
    s_idx = 0
    A, B = SCENARIOS["S1 Base"]
    again = pd.DataFrame([run_rep(s_idx, r, A, B) for r in range(min(10, args.reps))])
    first = raw[raw.escenario == "S1 Base"].head(len(again)).reset_index(drop=True)
    identical = bool(first.equals(again))
    sha = hashlib.sha256(open(os.path.join(out, "resultados_crudos.csv"), "rb").read()).hexdigest()
    with open(os.path.join(out, "entorno_y_reproducibilidad.txt"), "w", encoding="utf-8") as f:
        f.write(f"Python {sys.version.split()[0]} | {platform.platform()}\n")
        f.write(f"numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__} | matplotlib {matplotlib.__version__}\n")
        f.write(f"SEED_BASE = {SEED_BASE}\nreplicas por escenario = {args.reps}\n")
        f.write(f"Re-ejecucion de {len(again)} replicas de S1 identica bit a bit: {identical}\n")
        f.write(f"SHA-256 de resultados_crudos.csv: {sha}\n")
        f.write(f"Tiempo total: {time.time() - t0:.1f} s\n")
    print(f"Reproducible (bit a bit): {identical} | SHA-256: {sha[:16]}... | total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
