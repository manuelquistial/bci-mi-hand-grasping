# SimOpenBCI

Simulador que envía señales EEG multicanales a **Lab Streaming Layer (LSL)** leyendo datos desde archivo.

Arquitectura **Ports & Adapters**: el núcleo (`PlaybackService`) no depende de Qt ni de pylsl; esos son adapters. Los formatos de dataset viven en `dataset_loaders`.

Gestión de entorno y dependencias con **[uv](https://docs.astral.sh/uv/)**.

Más detalle por dataset y extensión: [`docs/`](docs/README.md).

## Qué hace el código

- **`simopenbci.PlaybackService`**: lee muestras con un loader y las empuja a un `SampleSink` con timing de reloj compensado.
- **`simopenbci.cli`**: argparse y orquestación (misma CLI pública).
- **`dataset_loaders`**: cargadores intercambiables (BCI Competition IV 2a, PhysioNet EEGBCI, CSV genérico), con detección por ruta y multi-sujeto.
- **Entrypoints**: `uv run simopenbci …` o `uv run python main.py …` (por defecto: primer sujeto BCI IV 2a).

## Setup con uv

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) y Python 3.11+ (el proyecto fija `3.11` en `.python-version`).

```bash
cd /Users/manueljurado/implementacion_articulos/simulator

# Crea .venv, instala deps de runtime + grupo dev (pytest) y el paquete en editable
uv sync --all-groups
```

Comandos habituales (no hace falta activar el venv a mano):

```bash
uv run simopenbci --help
uv run pytest -q
uv run python main.py --no-dialog
```

Si prefieres el shell activado:

```bash
source .venv/bin/activate
simopenbci --help
pytest -q
```

### Añadir dependencias

```bash
uv add nombre-paquete          # runtime → pyproject.toml + uv.lock
uv add --dev nombre-paquete    # grupo dev
uv sync --all-groups           # re-sincronizar tras cambios
```

### macOS — liblsl

`pylsl` necesita la biblioteca nativa **liblsl**:

```bash
brew install labstreaminglayer/tap/lsl
PYTHON_SITE=$(uv run python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$PYTHON_SITE/pylsl/lib"
cp "$(brew --prefix lsl)"/lib/liblsl*.dylib "$PYTHON_SITE/pylsl/lib/"
```

## Datasets soportados

### BCI Competition IV Dataset 2a

- **Ruta típica**: carpeta con `BCI_Competition_IV2a` en el path (ej. `data/sujetos_BCI_Competition_IV2a/`).
- **Sujetos**: A01–A09; **runs**: T (training), E (evaluation). Archivos: `A01T.txt`, `A01E.txt`, etc.
- **Formato**: una línea por muestra; columna 0 = tiempo, columnas 1–22 = 22 canales EEG.
- **Metadatos**: 22 canales, 250 Hz, stream `BCI_IV2a` (nombres de canal en el StreamInfo LSL).
- Ver [`docs/bci-iv2a-dataset.md`](docs/bci-iv2a-dataset.md).

### PhysioNet EEGBCI (mismo que articulo_1)

- **Dataset**: EEG Motor Movement/Imagery, 8 canales, 160 Hz.
- **Uso**: `--dataset physionet_eegbci`, `--subject 1` o `--subject S001`.
- Ver [`docs/physionet.md`](docs/physionet.md).

### CSV/TXT genérico

- Líneas que empiezan con `%` se ignoran; columnas 1–8 por defecto = 8 canales.
- Stream `BioSemi`, 8 canales, 250 Hz.

### Añadir nuevos datasets

1. Heredar de `dataset_loaders.base.BaseDatasetLoader` (`metadata`, `iter_samples`).
2. Opcional: `detect_from_path()`, `default_data_root()`, `list_subjects` / `resolve_subject_file`.
3. Registrar en `dataset_loaders/__init__.py` → `LOADERS`.

Ver [`docs/multi-subject.md`](docs/multi-subject.md) para multi-sujeto.

## Marcadores de trial (LSL Markers)

Si junto al TXT existe un sidecar ``<archivo>.events.csv``, el simulador abre un
segundo stream LSL ``<eeg_name>_Markers`` y emite un marcador en el instante de
cada evento (p. ej. inicio de trial).

Formato CSV:

```csv
sample_index,code,label,kind
1200,T1,left,trial_start
1840,T0,rest,rest
```

- **PhysioNet**: exportar TXT+events con
  `uv run python scripts/export_physionet_with_events.py --edf-dir … --subject 1`
  (`T1`/`T2` = trial_start left/right; `T0` = rest).
- **BCI IV 2a**: generar el sidecar desde los eventos GDF (768/769–772) al exportar.

```bash
uv run simopenbci --dataset physionet_eegbci --subject S001 \
  --data-root data/physionet_eegbci --lsl-name OpenBCI_EEG --no-dialog
```

Desactivar marcadores: `--no-markers`. Ruta explícita: `--events path/to/file.events.csv`.

### Sesión grabada por el pipeline

El pipeline escribe la carpeta (`signal.f32`, `time.f64`, `markers.jsonl`, `session.json`). Este simulador solo la lee:

```bash
uv run simopenbci --dataset recorded_session --file /ruta/a/la/sesion \
  --lsl-name OpenBCI_EEG --no-dialog
```

Otros datasets (BCI IV 2a, PhysioNet, CSV) siguen en sus loaders.

## Uso por línea de comandos

Sin argumentos, streamea el **primer sujeto** de BCI IV 2a desde `sample_data/bci_iv2a/` (o `data/` si existe):

```bash
uv run simopenbci
# equivalente headless:
uv run simopenbci --no-dialog
```

```bash
# PhysioNet (primer sujeto en sample_data/physionet_eegbci)
uv run simopenbci --dataset physionet_eegbci --no-dialog

# Sujeto / run concreto
uv run simopenbci --dataset bci_iv2a --subject A01 --run T --channels 8 --no-dialog
uv run simopenbci --dataset bci_iv2a --subject A09 --run E --lsl-name OpenBCI_EEG --no-dialog

# Listar sujetos
uv run simopenbci --dataset bci_iv2a --list-subjects

# Solo si quieres elegir un archivo a mano
uv run simopenbci --pick-file
```

## Requisitos

- **uv** + **Python 3.11+**
- **PyQt5**: solo cuando hay diálogos (sin `--no-dialog`).
- **pylsl** (+ **liblsl** en macOS).
- **pytest**: grupo `dev` (`uv sync --all-groups`).

### Formato genérico (ejemplo)

```
%,header,optional
1, 12.3, -4.5, 0.1, 2.2, 3.3, 4.4, 5.5, 6.6
2, 12.4, -4.4, 0.2, 2.3, 3.4, 4.5, 5.6, 6.7
```

## Tests

```bash
uv sync --all-groups
uv run pytest -q
```

Los tests usan `RecordingSink` (sin LSL). El test de `LslSampleSink` se omite si liblsl no está disponible.

## Conexión con articulo_1

```bash
uv run simopenbci --dataset bci_iv2a --file data/sujetos_BCI_Competition_IV2a/A01T.txt \
  --channels 8 --lsl-name OpenBCI_EEG --no-dialog
```

En articulo_1:

```bash
python scripts/run_openbci_lsl.py --model models/mi_mdmf.joblib --lsl-name OpenBCI_EEG --use-stream-order
```

Ver [CONEXION_SIMULADOR.md](../articulo_1/CONEXION_SIMULADOR.md).

## Estructura del proyecto

```
simulator/
├── main.py                    # Entrypoint CLI
├── simopenbci/                # Núcleo Ports & Adapters
│   ├── playback.py            # PlaybackService
│   ├── config.py              # PlaybackConfig
│   ├── ports.py               # SampleSink, Notifier, Clock
│   ├── clock.py               # RealtimeClock, InstantClock
│   ├── sinks/                 # LslSampleSink, RecordingSink
│   ├── notifiers/             # ConsoleNotifier, QtNotifier
│   ├── gui.py                 # Qt file picker (lazy)
│   └── cli.py                 # argparse + orquestación
├── dataset_loaders/           # Loaders extensibles
│   ├── base.py
│   ├── parsing.py             # Parseo CSV/TXT compartido
│   ├── bci_competition_iv2a.py
│   ├── physionet_eegbci.py
│   └── generic_csv.py
├── sample_data/               # Datasets de ejemplo (BCI IV 2a, PhysioNet)
│   ├── bci_iv2a/
│   └── physionet_eegbci/
├── docs/                      # Guías (datasets, multi-sujeto)
│   ├── README.md
│   ├── bci-iv2a-dataset.md
│   ├── physionet.md
│   └── multi-subject.md
├── tests/                     # pytest + fixtures
├── pyproject.toml             # proyecto + deps (uv)
├── uv.lock                    # lockfile reproducible
├── .python-version            # 3.11
└── README.md
```
