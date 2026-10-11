# -*- coding: utf-8 -*-
"""
Cálculos reproducibles de la Guía de evidencia PBM (jans-fromow.github.io/pbm-retina-evidencia)
Autor: Jans Fromow-Guerra · octubre de 2026 · Python 3 + numpy + scipy + statsmodels

Todos los datos de entrada provienen de los artículos publicados (página indicada en cada bloque).
Son ilustraciones de planeación: suponen observaciones independientes (ojos), pruebas bilaterales,
proporciones con la fórmula de Fleiss con corrección de continuidad y medias con t para muestras
independientes. No reproducen los modelos de medidas repetidas de los ensayos.
Ejecutar:  python3 calculos_pbm.py
"""
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

TT = TTestIndPower()

def n_fleiss(p_ctrl, p_trt, alpha, power, k=1.0, cc=True):
    """Ojos en el grupo control y total, con k = n_trt / n_ctrl. Fleiss (1981), con corrección de continuidad (Fleiss-Tytun-Ury 1980)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p_ctrl + k * p_trt) / (1 + k)
    d = abs(p_ctrl - p_trt)
    nc = (za * np.sqrt(pbar * (1 - pbar) * (1 + 1 / k)) + zb * np.sqrt(p_ctrl * (1 - p_ctrl) + p_trt * (1 - p_trt) / k)) ** 2 / d ** 2
    if cc:
        nc = nc / 4 * (1 + np.sqrt(1 + 2 * (k + 1) / (k * nc * d))) ** 2
    # convención: cada grupo se redondea hacia arriba; el total es la suma de los grupos
    return int(np.ceil(nc)), int(np.ceil(nc) + np.ceil(k * nc))

def fisher(a, n1, b, n2):
    return stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]

def exact_power(n1, p1, n2, p2, alpha=0.05):
    """Poder exacto de la prueba de Fisher bilateral por enumeración; y razón de exageración (error tipo M) entre resultados significativos."""
    f1 = stats.binom.pmf(range(n1 + 1), n1, p1); f2 = stats.binom.pmf(range(n2 + 1), n2, p2)
    pw = 0.0; num = 0.0
    for a in range(n1 + 1):
        if f1[a] < 1e-12: continue
        for b in range(n2 + 1):
            if f2[b] < 1e-12: continue
            if fisher(a, n1, b, n2) < alpha:
                w = f1[a] * f2[b]; pw += w; num += w * abs(a / n1 - b / n2)
    return pw, (num / pw) / abs(p1 - p2)

print("=" * 72); print("LIGHTSITE III · agudeza visual (Boyer 2024, p. 489 y Tabla 2, p. 492)")
sd = np.sqrt(((91 - 1) * 9.16 ** 2 + (54 - 1) * 8.30 ** 2) / (91 + 54 - 2))
print(f"DE combinada observada: {sd:.2f} letras")
for d in (3.18, 2.4):
    print(f"Poder para {d} letras, 91 vs 54, α 0.025: {TT.power(d / sd, nobs1=91, ratio=54 / 91, alpha=0.025):.0%}")
for pw in (0.8, 0.9):
    n1 = TT.solve_power(3 / sd, power=pw, ratio=0.5, alpha=0.025)
    print(f"Ojos para 3 letras, 2:1, α 0.025, poder {pw:.0%}: {np.ceil(n1) + np.ceil(n1 / 2):.0f}")
print(f"Error estándar de la diferencia entre grupos (DE {sd:.2f}, 91 vs 54): {sd * np.sqrt(1 / 91 + 1 / 54):.2f} letras (sin correlación entre ojos)")

print("=" * 72); print("LIGHTSITE III · atrofia geográfica incidente (Jaffe 2026, Tabla 3, p. 789)")
print(f"Fisher por ojo 6/87 vs 12/50: p = {fisher(6, 87, 12, 50):.4f}")
for lab, pc, pt, a, pw in [("Observado 24.0% vs 6.8%, α 0.05, 80%", .24, .068, .05, .8),
                           ("Mismo efecto, α 0.025, 80%", .24, .068, .025, .8),
                           ("Mismo efecto, Bonferroni 17 pruebas, 80%", .24, .068, .05 / 17, .8),
                           ("Control 12%, RRR 50%, α 0.05, 80%", .12, .06, .05, .8),
                           ("Control 12%, RRR 30%, α 0.05, 80%", .12, .084, .05, .8)]:
    print(f"{lab}: {n_fleiss(pc, pt, a, pw, k=2.0)[1]} ojos (2:1)")
_, t = n_fleiss(.24, .068, .025, .9, k=2.0)
print(f"Mismo efecto, α 0.025, 90%: {t} ojos; × efecto de diseño 1.24 / 0.80 (20% de abandono): {np.ceil(t * 1.24 / 0.80):.0f} ojos")
pw, m = exact_power(50, .12, 87, .06)
print(f"Poder exacto (Fisher) para 12% vs 6% con 50 y 87 ojos: {pw:.1%}; exageración media entre significativos: {m:.1f}×")
print("Índice de fragilidad (sumando eventos al grupo PBM):")
for e in range(7, 11): print(f"  PBM {e}/87 vs 12/50: p = {fisher(e, 87, 12, 50):.3f}")
print(f"Sensibilidad: sham 9/50 vs 6/87: p = {fisher(6, 87, 9, 50):.3f}")
print(f"Excluyendo el ojo con enfermedad de Best (FDA): 11/49 vs 6/87: p = {fisher(6, 87, 11, 49):.3f}")

print("=" * 72); print("Multiplicidad (ilustración: independencia y todas las nulas ciertas)")
for k in (1, 10, 17, 18, 27): print(f"k = {k}: P(≥1 falso positivo) = {1 - 0.95 ** k:.0%}; Bonferroni {0.05 / k:.4f}")
tabla3 = {'Déficit baja luminancia': 0.0003, 'AG M24': 0.007, 'VFQ actividades': 0.0078, 'VFQ compuesto': 0.0182, 'AG M13': 0.024,
          'VFQ visitas': 0.03, 'VFQ periódicos': 0.038, 'VFQ cerca': 0.0503, 'Velocidad lectura': 0.12, 'LLBCVA': 0.29,
          'Área AG raíz': 0.52, 'Área AG': 0.63, 'Drusas': 0.65, 'CPS': 0.73, 'Agudeza lectura': 0.78, 'Sensibilidad contraste': 0.82, 'Otra (≥0.05)': 0.9}
print("Holm sobre las 17 pruebas de la Tabla 3:")
for i, (k_, p) in enumerate(sorted(tabla3.items(), key=lambda x: x[1])):
    thr = 0.05 / (17 - i); ok = p < thr
    print(f"  {i + 1:2d}. {k_:26s} p = {p:<7} umbral {thr:.4f} {'pasa' if ok else 'no pasa → se detiene'}")
    if not ok: break

print("=" * 72); print("EYE-LIGHT / DRUSEN (Iovino 2026, pp. 3-4) · 1:1, por grupo")
for d, a in ((3.75, .025), (2.0, .025)):
    print(f"Δ {d} letras, DE 7, α {a}: 80% {np.ceil(TT.solve_power(d / 7, power=.8, alpha=a)):.0f} · 90% {np.ceil(TT.solve_power(d / 7, power=.9, alpha=a)):.0f}")
print(f"Poder para 3.75 letras con 72 vs 54, α 0.025: {TT.power(3.75 / 7, nobs1=72, ratio=54 / 72, alpha=0.025):.0%}")
print(f"Poder para 2 letras con 72 vs 54, α 0.025: {TT.power(2 / 7, nobs1=72, ratio=54 / 72, alpha=0.025):.0%}")
for lab, pc, pt in [("≥ 5 letras 8.9% → 15%", .089, .15), ("MNV 6.3% → 3.15%", .063, .0315), ("DMRE tardía 15% → 7.5% (24 m)", .15, .075)]:
    print(f"{lab}, α 0.05: 80% {n_fleiss(pc, pt, .05, .8)[0]} · 90% {n_fleiss(pc, pt, .05, .9)[0]}")
pw, _ = exact_power(64, .063, 74, .0315)
print(f"Poder exacto para MNV 6.3% vs 3.15% con 64 y 74 ojos: {pw:.1%}")
print(f"Fragilidad MNV: 4/64 vs 1/74: p = {fisher(4, 64, 1, 74):.2f}")

print("=" * 72); print("El ensayo que haría falta · 1:1, α 0.05, un ojo por sujeto (sujetos totales)")
for lab, pc, pt in [("DMRE tardía 15% → 7.5%", .15, .075), ("DMRE tardía 15% → 10%", .15, .10), ("AG 12% → 6%", .12, .06),
                    ("≥ 10 letras 15.1% → 25.1%", .151, .251), ("≥ 10 letras 15.1% → 20.1%", .151, .201)]:
    print(f"{lab}: 80% {n_fleiss(pc, pt, .05, .8)[1]} · 90% {n_fleiss(pc, pt, .05, .9)[1]}")
