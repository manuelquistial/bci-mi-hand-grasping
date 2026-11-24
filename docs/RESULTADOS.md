# Resultados y análisis — BCI motor imagery

Estado del proyecto hasta noviembre 2026. Artefactos principales en `pipeline/batch-results/` y packs online en `pipeline/online-models/`.

## Resumen ejecutivo

El repositorio es un **BCI de imaginación motora (mano izquierda/derecha)** con tres paquetes: `simulator` → LSL → `pipeline` → HTTP/WebSocket → `game`.

| Dataset | Hallazgo principal |
|---------|-------------------|
| **Shuqfa 2024** (103 sujetos sanos) | Señal usable. Mejor BA nested ≈ **75%**. Pack LOSO hold-out S001: offline **72%**, online **79%** (11/14 trials). |
| **Liu 2024 stroke** (50 pacientes) | Decoding cerca del azar. Mejor BA nested ≈ **57%**; CSP+euclidean+LR ≈ **51%**. |

Pipeline con más evidencia: **alineación euclidiana + CSP + logistic regression / linear SVM**.

---

## Arquitectura (contexto)

```
simulator ──LSL (EEG + Markers)──▶ pipeline ──HTTP :8765──────▶ game
                                      │      ──control WS :8090─▶ game (/game)
                                      └──────  observe WS :8091─▶ game (/monitor)
```

Detalle en [`ARCHITECTURE.md`](ARCHITECTURE.md).

**Ventaneo online:** cinco ventanas de 2 s con paso 0,5 s (0–2 … 2–4 s desde el inicio de la imaginación).

---

## 1. Offline nested — grid `c7-full`

### Protocolo

- Nested CV cross-subject: fold externo **5 grupos por sujeto** (esta corrida usó `nested`, no `nested_loso`).
- Fold interno ≈ 3: hiperparámetros con balanced accuracy promediada por persona.
- Filtro: notch 50 Hz + banda **8–30 Hz**.
- Espacial: `none`, `car`, `hjorth`, `large_laplacian`, `spherical_spline`.
- Alineación: `none` o `euclidean`.
- Features: `ts`, `tdp`, `psd`, `csp`, `fbcsp`, `covariance`.
- ~**1300** combinaciones; datasets no mezclados.

### Artefactos

| Archivo | Contenido |
|---------|-----------|
| `pipeline/batch-results/c7-full-comparison.csv` | Ranking (1310 filas de datos) |
| `pipeline/batch-results/c7-full-report.csv` | Reporte detallado |
| `pipeline/batch-results/c7-full.json` | Manifiesto completo (~390 MB) |
| `pipeline/batch-results/c7-full.log` | Log de corrida |
| `pipeline/batch-results/c7-full-flow-report.pdf` | Diagrama de flujo |

### Shuqfa 2024 — mejores resultados

| Rank | Spatial | Alignment | Features | Classifier | BA | Acc | κ |
|------|---------|-----------|----------|------------|-----|-----|---|
| 1 | none | euclidean | csp | logistic_regression | **0.7501** | 0.7503 | 0.5007 |
| 2 | none | euclidean | csp | linear_svm | 0.7472 | 0.7477 | 0.4954 |
| 3 | car | euclidean | csp | logistic_regression | 0.7444 | 0.7444 | 0.4888 |
| 4 | car | euclidean | csp | slda | 0.7443 | 0.7444 | 0.4888 |
| 5 | none | euclidean | csp | lda | 0.7433 | 0.7437 | 0.4875 |

**Mejor BA aproximado por feature (Shuqfa):** CSP 0.750 · TDP 0.686 · covariance 0.683 · FBCSP 0.658 · PSD 0.648 · TS 0.556.

### Liu 2024 stroke — mejores resultados

| Rank | Spatial | Alignment | Features | Classifier | BA | Acc | κ |
|------|---------|-----------|----------|------------|-----|-----|---|
| 1 | spherical_spline | none | ts | random_subspace_knn | **0.5677** | 0.5674 | 0.1349 |
| 2 | spherical_spline | euclidean | ts | random_forest | 0.5644 | 0.5644 | 0.1286 |
| 3 | car | none | psd | bagged_trees | 0.5627 | 0.5624 | 0.1247 |
| — | none | euclidean | csp | logistic_regression | ~0.515 | ~0.515 | ~0.03 |

Liu permanece cerca del azar; el stack que gana en Shuqfa no transfiere.

---

## 2. Comparaciones paper (MDM / TWFB)

| Artefacto | Hallazgo |
|-----------|----------|
| `pipeline/batch-results/paper-csp-comparison.csv` | Shuqfa CSP+MDM+euclidean: BA hasta ~**0.72** (mejor ~0.723 con large_laplacian/euclidean MDM). Liu CSP+MDM/TWFB ≈ **0.48–0.52**. |
| `pipeline/batch-results/paper-full-comparison.csv` | Covarianza MDM: Liu BA **0.504**; Shuqfa BA **0.534**. TWFB ≈ chance. |

---

## 3. Online / LOSO — packs congelados (Shuqfa S001)

Config: `pipeline/configs/fsb-euclidean-csp.json`  
Script: `pipeline/scripts/run_fsb_euclidean_csp.sh`  
Slurm: `pipeline/scripts/slurm_euclidean_csp.sbatch`  
Índice: `pipeline/online-models/index.json`

### Hold-out LOSO (evaluación limpia)

- Entrenamiento: **102** sujetos; **S001** fuera del train.
- Carpeta: `pipeline/batch-results/fsb-euclidean-csp-loso-20261110-165310/`
- Log Slurm: `pipeline/batch-results/slurm-8293144.out`

| Pack / head | Offline Acc | Online Acc | Trials |
|-------------|-------------|------------|--------|
| `20261110-165457-euclidean-csp-linear_svm` | **0.722** | **0.786** (11/14) | 14 |
| `20261110-165405-euclidean-csp-logistic_regression` | **0.722** | **0.786** (11/14) | 14 |

Replay adicional: `pipeline/batch-results/online-replay-20261112-202921.jsonl` → Acc final **0.7857** (linear_svm).

### Sin hold-out (optimista — no usar como claim)

- Entrenamiento: **103** sujetos (incluye el sujeto online).
- Carpeta: `pipeline/batch-results/fsb-euclidean-csp-20261110-164110/`

| Pack / head | Offline Acc | Online Acc | Trials |
|-------------|-------------|------------|--------|
| `20261110-164327-euclidean-csp-linear_svm` | 0.727 | **0.857** (12/14) | 14 |
| `20261110-164229-euclidean-csp-logistic_regression` | 0.724 | **0.857** (12/14) | 14 |

El juego debe preferir packs con `held_out` / `losoOnly` (ver `game/src/lib/pipeline/catalog.ts`).

---

## 4. Datasets

| ID | Docs | Especificación usada |
|----|------|----------------------|
| **shuqfa2024** | `simulator/docs/shuqfa2024.md` | PhysioNet EEGBCI limpio; **103** sujetos; run **02**; **64 ch @ 160 Hz**; L/R ~4 s |
| **liu2024_stroke** | `simulator/docs/liu2024-stroke-mi.md` | Grasp MI en stroke; **50** pacientes; **40** trials L/R; **29 ch @ 500 Hz**; 4 s |
| **recorded_session** | pipeline / simulator READMEs | Cyton **8 ch @ 250 Hz** + EMG antebrazo **8 ch** |

---

## 5. Interpretación

1. **Pipeline elegido con evidencia:** euclidean + CSP + LR/SVM. Coherente entre nested grid, packs FSB y replay online.
2. **Shuqfa es el dataset operativo** para demo/juego cross-subject.
3. **Liu no se decodifica bien** con este stack; el cuello de botella parece de dominio (población stroke, SNR, montaje), no solo de búsqueda incompleta.
4. **78.6% online es alentador pero frágil:** 14 trials de un solo sujeto. El 85.7% sin hold-out no es evaluación limpia.
5. **El ganador nested describe estos datos**; no sustituye una cohorte independiente nueva.

---

## 6. Estado del trabajo

### Completado

- Sistema de tres paquetes + contratos + CI de arquitectura.
- Loaders Shuqfa / Liu / sesión grabada.
- Pre-model offline (filtro → espacial → alineación → features → clasificador → métricas).
- Batch nested masivo (`c7-full-*`) y corridas paper MDM/TWFB.
- Export cross-subject, registry de modelos, `online-replay` sin LSL, scripts Slurm FSB.
- Game: calibración offline, online simulado/real, control WS desde packs LOSO, `/monitor`.

### Pendiente / límites

| Ítem | Nota |
|------|------|
| Checklist E2E | `scripts/e2e_checklist.md` aún sin cerrar |
| `nested_loso` full grid | Código lo soporta; `c7-full` usó nested 5-fold |
| Liu | Problema científico/ingenieril abierto |
| Thesis write-up | No hay PDF/notebooks de tesis en el repo |
| Deuda arquitectura | Algunos `application/` aún importan `adapters.*` concretos |

---

## 7. Números para citar rápido

| Escenario | Métrica | Valor |
|-----------|---------|-------|
| Offline nested Shuqfa (mejor) | BA | **0.750** |
| Offline nested Liu (mejor) | BA | **0.568** |
| LOSO hold-out S001 offline | Acc | **0.722** |
| LOSO hold-out S001 online replay | Acc | **0.786** (11/14) |
| Sin hold-out online (fuga) | Acc | 0.857 (12/14) — no reportar como evaluación limpia |

---

## Referencias de archivos

- Arquitectura: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Pipeline: [`pipeline/README.md`](../pipeline/README.md)
- Simulador: [`simulator/README.md`](../simulator/README.md)
- Resultados batch: `pipeline/batch-results/`
- Modelos online: `pipeline/online-models/`
