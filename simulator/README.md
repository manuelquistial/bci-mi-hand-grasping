# SimOpenBCI

Simulador que envía señales EEG multicanales a **Lab Streaming Layer (LSL)** leyendo datos desde archivo.

Arquitectura **Ports & Adapters**: el núcleo (`PlaybackService`) no depende de Qt ni de pylsl; esos son adapters. Los formatos de dataset viven en `dataset_loaders`.

Gestión de entorno y dependencias con **[uv](https://docs.astral.sh/uv/)**.

Más detalle por dataset y extensión: [`docs/`](docs/README.md).

## Qué hace el código

- **`simopenbci.PlaybackService`**: lee muestras con un loader y las empuja a un `SampleSink` con timing de reloj compensado.
- **`simopenbci.cli`**: argparse y orquestación (misma CLI pública).
- **`dataset_loaders`**: cargadores intercambiables (Shuqfa 2024, Liu 2024), con detección por ruta y multi-sujeto.
- **Entrypoints**: `uv run simopenbci …` o `uv run python main.py …` (por defecto: registro 02 del primer sujeto de Shuqfa 2024).

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

### Liu et al. 2024 (imaginación de agarre, ACV)

- **Dataset**: 50 pacientes, 40 ensayos izquierda/derecha, 500 Hz.
- **Uso**: `--dataset liu2024_stroke`, `--subject 1` o `--subject sub-01`.
- Ver [`docs/liu2024-stroke-mi.md`](docs/liu2024-stroke-mi.md).

### Shuqfa et al. 2024 (PhysioNet EEG Motor Movement/Imagery)

- **Dataset**: el eegmmidb de PhysioNet, depurado: 103 sujetos, 12 registros de tarea, 64 canales, 160 Hz. Un CSV por registro.
- **Uso**: `--dataset shuqfa2024`, `--subject 1`, `--run 2`. Sin `--run` se envía el registro `02` (imaginación unilateral: reposo, izquierda, derecha).
- Ver [`docs/shuqfa2024.md`](docs/shuqfa2024.md).

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

- **Shuqfa 2024**: los marcadores salen del CSV de anotaciones del mismo registro.

```bash
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 \
  --lsl-name OpenBCI_EEG --no-dialog
```

Desactivar marcadores: `--no-markers`. Ruta explícita: `--events path/to/file.events.csv`.

### Sesión grabada por el pipeline

El pipeline escribe la carpeta (`signal.f32`, `time.f64`, `markers.jsonl`, `session.json`). Este simulador solo la lee:

```bash
uv run simopenbci --dataset recorded_session --file /ruta/a/la/sesion \
  --lsl-name OpenBCI_EEG --no-dialog
```

Los demás datasets siguen en sus loaders.

## Uso por línea de comandos

Sin argumentos, streamea el registro `02` del primer sujeto de Shuqfa 2024 desde `data/shuqfa2024/`:

```bash
uv run simopenbci
# equivalente headless:
uv run simopenbci --no-dialog
```

```bash
# PhysioNet depurado (Shuqfa 2024), registro de imaginación unilateral
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 --no-dialog

# Sujeto / run concreto
uv run simopenbci --dataset liu2024_stroke --subject sub-01 --no-dialog

# Listar sujetos
uv run simopenbci --dataset shuqfa2024 --list-subjects

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
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 \
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
│   ├── liu2024_stroke_mi.py
│   ├── shuqfa2024.py
│   ├── edf.py
│   └── generic_csv.py
├── data/                      # Datasets locales (gitignored): data/<id>/
│   ├── liu2024_stroke/
│   └── shuqfa2024/
├── docs/                      # Guías (datasets, multi-sujeto)
│   ├── README.md
│   ├── liu2024-stroke-mi.md
│   ├── shuqfa2024.md
│   └── multi-subject.md
├── tests/                     # pytest + fixtures
├── pyproject.toml             # proyecto + deps (uv)
├── uv.lock                    # lockfile reproducible
├── .python-version            # 3.11
└── README.md
```
