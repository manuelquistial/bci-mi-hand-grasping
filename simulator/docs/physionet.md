# PhysioNet EEGBCI en el simulador

Permite enviar por LSL el **mismo dataset** que usa articulo_1 para entrenar (PhysioNet EEG Motor Movement/Imagery: 8 canales, 160 Hz, 3 clases rest/left/right).

## Dataset

- **PhysioNet**: EEG Motor Movement/Imagery ([eegmmidb](https://physionet.org/content/eegmmidb)).
- **En articulo_1**: preset `physionet_eegbci`, runs **4, 8, 12** (motor imagery left/right), **primeros 8 canales**, **160 Hz**.

## Pasos

### 1. Exportar desde articulo_1 a TXT

En el proyecto **articulo_1**:

```bash
cd articulo_1
source .venv/bin/activate
# Exportar sujetos 1, 2, 3 a la carpeta del simulador
python scripts/export_physionet_to_simulator.py --subjects 1 2 3 --out ../simulator/data/physionet_eegbci/
```

Se crean `S001.txt`, `S002.txt`, `S003.txt` (8 columnas, una muestra por línea, 160 Hz).

### 2. Stream desde el simulador

En el proyecto **simulator**:

```bash
cd simulator
uv sync --all-groups
# Listar sujetos exportados
uv run simopenbci --dataset physionet_eegbci --list-subjects

# Stream sujeto 1 (mismo formato que espera el pipeline)
uv run simopenbci --dataset physionet_eegbci --subject 1 --lsl-name OpenBCI_EEG --no-dialog
# O con --max-samples para prueba
uv run simopenbci --dataset physionet_eegbci --subject S001 --lsl-name OpenBCI_EEG --max-samples 2000 --no-dialog
```

### 3. Procesar en articulo_1

El pipeline está entrenado a **160 Hz** y **8 canales**; el stream PhysioNet es 160 Hz y 8 ch, así que **no se aplica resampling**. Conectar igual que con BCI IV 2a:

```bash
cd articulo_1
python scripts/run_openbci_lsl.py --model models/mi_mdmf.joblib --lsl-name OpenBCI_EEG --use-stream-order
```

## Formato TXT

- Una línea por muestra.
- Ocho valores separados por coma (canales 1–8).
- Sin cabecera (o líneas que empiecen por `%` se ignoran).
- Frecuencia: 160 Hz (intervalo 1/160 s entre muestras).
