"""
hmm_markov.py
=============
Implementación desde cero (solo NumPy/SciPy) de:

* ``HMM``          : Modelo Oculto de Markov discreto con
                     - muestreo (simulación),
                     - algoritmo Forward con escalado numérico,
                     - algoritmo Backward,
                     - algoritmo de Viterbi (espacio logarítmico),
                     - algoritmo de Baum-Welch (EM) con varias secuencias y reinicios.
* ``MarkovChain``  : Cadena de Markov visible de primer orden sobre las observaciones.
* Utilidades       : distribución estacionaria, tiempos medios hasta absorción,
                     alineación de estados (problema de permutación de etiquetas).

Notación (Rabiner, 1989):
    S = {s_1..s_N}   estados ocultos
    V = {v_1..v_M}   símbolos observables
    A[i, j] = P(q_{t+1}=s_j | q_t=s_i)     matriz de transición
    B[j, k] = P(o_t=v_k | q_t=s_j)         matriz de emisión
    pi[i]   = P(q_1=s_i)                   distribución inicial
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _check_stochastic(M: np.ndarray, name: str) -> None:
    """Verifica que cada fila sea una distribución de probabilidad válida."""
    if np.any(M < -EPS):
        raise ValueError(f"{name} contiene probabilidades negativas")
    s = M.sum(axis=-1)
    if not np.allclose(s, 1.0, atol=1e-6):
        raise ValueError(f"{name}: las filas deben sumar 1 (suman {s})")


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Distribución estacionaria pi_inf tal que pi_inf A = pi_inf."""
    w, v = np.linalg.eig(A.T)
    k = np.argmin(np.abs(w - 1.0))
    p = np.real(v[:, k])
    p = np.abs(p)
    return p / p.sum()


def mean_hitting_time(A: np.ndarray, target: int) -> np.ndarray:
    """
    Tiempo medio (en pasos) para alcanzar el estado ``target`` partiendo de
    cada estado. Se resuelve (I - Q) h = 1 donde Q es A sin la fila/columna target.
    """
    N = A.shape[0]
    idx = [i for i in range(N) if i != target]
    Q = A[np.ix_(idx, idx)]
    h = np.linalg.solve(np.eye(len(idx)) - Q, np.ones(len(idx)))
    out = np.zeros(N)
    out[idx] = h
    return out


def align_states(B_true: np.ndarray, B_learn: np.ndarray) -> np.ndarray:
    """
    Resuelve la ambigüedad de etiquetas de un HMM aprendido (los estados ocultos
    pueden aparecer permutados). Devuelve ``perm`` tal que el estado verdadero i
    corresponde al estado aprendido perm[i] (asignación húmgara sobre la distancia
    L1 entre filas de B).
    """
    cost = np.abs(B_true[:, None, :] - B_learn[None, :, :]).sum(-1)
    _, cols = linear_sum_assignment(cost)
    return cols


# --------------------------------------------------------------------------- #
# Modelo Oculto de Markov
# --------------------------------------------------------------------------- #
class HMM:
    """HMM discreto (emisiones categóricas)."""

    def __init__(self, A, B, pi):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.N, self.M = self.B.shape
        _check_stochastic(self.A, "A")
        _check_stochastic(self.B, "B")
        _check_stochastic(self.pi, "pi")

    # ---------------------------- simulación ------------------------------- #
    def sample(self, T: int, rng: np.random.Generator):
        """Genera (estados, observaciones) de longitud T por muestreo inverso."""
        cumA = np.cumsum(self.A, axis=1)
        cumB = np.cumsum(self.B, axis=1)
        cumpi = np.cumsum(self.pi)
        u = rng.random((T + 1, 2))
        states = np.empty(T, dtype=int)
        obs = np.empty(T, dtype=int)
        s = min(int(np.searchsorted(cumpi, u[0, 0], side="right")), self.N - 1)
        for t in range(T):
            states[t] = s
            obs[t] = min(int(np.searchsorted(cumB[s], u[t + 1, 1], side="right")), self.M - 1)
            s = min(int(np.searchsorted(cumA[s], u[t + 1, 0], side="right")), self.N - 1)
        return states, obs

    # ------------------------------ Forward -------------------------------- #
    def forward(self, obs):
        """
        Algoritmo Forward con escalado.
        Devuelve alpha escalado (T,N), factores de escala c (T,) y log P(O | lambda).
        """
        T = len(obs)
        alpha = np.empty((T, self.N))
        c = np.empty(T)
        a = self.pi * self.B[:, obs[0]]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c, float(np.log(c).sum())

    def loglik(self, obs) -> float:
        return self.forward(obs)[2]

    # ------------------------------ Backward ------------------------------- #
    def backward(self, obs, c):
        T = len(obs)
        beta = np.empty((T, self.N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def posterior(self, obs):
        """Probabilidades suavizadas gamma_t(i) = P(q_t=s_i | O, lambda)."""
        alpha, c, _ = self.forward(obs)
        beta = self.backward(obs, c)
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    # ------------------------------ Viterbi -------------------------------- #
    def viterbi(self, obs):
        """Secuencia de estados más probable (programación dinámica en log)."""
        T = len(obs)
        with np.errstate(divide="ignore"):
            logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = np.empty((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            tmp = delta[t - 1][:, None] + logA
            psi[t] = tmp.argmax(axis=0)
            delta[t] = tmp.max(axis=0) + logB[:, obs[t]]
        path = np.empty(T, dtype=int)
        path[-1] = int(delta[-1].argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # ---------------------------- Baum-Welch ------------------------------- #
    @staticmethod
    def _random_init(N, M, rng):
        return HMM(
            rng.dirichlet(np.ones(N), size=N),
            rng.dirichlet(np.ones(M), size=N),
            rng.dirichlet(np.ones(N)),
        )

    @classmethod
    def baum_welch(cls, sequences, N, M, rng, n_iter=100, tol=1e-5, restarts=1):
        """
        Estimación por máxima verosimilitud (EM). ``sequences`` es una lista de
        arreglos de observaciones. Devuelve (mejor HMM, historia log-verosimilitud
        por símbolo, iteraciones usadas). Se conserva el reinicio con mayor
        verosimilitud final.
        """
        total_T = sum(len(o) for o in sequences)
        best = None
        for _ in range(restarts):
            model = cls._random_init(N, M, rng)
            hist = []
            prev = -np.inf
            for it in range(1, n_iter + 1):
                A_num = np.zeros((N, N))
                B_num = np.zeros((N, M))
                pi_num = np.zeros(N)
                ll = 0.0
                for obs in sequences:
                    alpha, c, l = model.forward(obs)
                    beta = model.backward(obs, c)
                    ll += l
                    gamma = alpha * beta
                    gamma /= gamma.sum(axis=1, keepdims=True)
                    W = model.B[:, obs[1:]].T * beta[1:] / c[1:, None]
                    A_num += model.A * (alpha[:-1].T @ W)
                    for k in range(M):
                        B_num[:, k] += gamma[obs == k].sum(axis=0)
                    pi_num += gamma[0]
                ll /= total_T
                hist.append(ll)
                A_new = A_num + 1e-6
                B_new = B_num + 1e-6
                model = cls(
                    A_new / A_new.sum(1, keepdims=True),
                    B_new / B_new.sum(1, keepdims=True),
                    (pi_num + 1e-6) / (pi_num + 1e-6).sum(),
                )
                if abs(ll - prev) < tol:
                    break
                prev = ll
            final = sum(model.loglik(o) for o in sequences) / total_T
            if best is None or final > best[1]:
                best = (model, final, hist, it)
        return best[0], best[2], best[3]


# --------------------------------------------------------------------------- #
# Cadena de Markov visible (sobre los símbolos observados)
# --------------------------------------------------------------------------- #
class MarkovChain:
    """Cadena de Markov de primer orden ajustada con suavizado de Laplace."""

    def __init__(self, P, p0):
        self.P = np.asarray(P, float)
        self.p0 = np.asarray(p0, float)
        _check_stochastic(self.P, "P")
        _check_stochastic(self.p0, "p0")

    @classmethod
    def fit(cls, seq, M, alpha=1.0):
        counts = np.full((M, M), alpha)
        np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
        p0 = np.bincount(seq, minlength=M) + alpha
        return cls(counts / counts.sum(1, keepdims=True), p0 / p0.sum())

    def loglik(self, seq) -> float:
        return float(np.log(self.p0[seq[0]]) + np.log(self.P[seq[:-1], seq[1:]]).sum())
