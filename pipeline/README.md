# pipeline

Pipeline EEG/BCI. El primer bloque **lee streams LSL** (Lab Streaming Layer).
Complementa al simulador `simulator/` (que publica EEG a LSL).

Gestión de entorno y dependencias con **[uv](https://docs.astral.sh/uv/)**.

## Setup

```bash
cd /Users/manueljurado/master_project/pipeline
uv sync --all-groups
```

En macOS/Linux, `pylsl` necesita la librería nativa **liblsl**. Si `import pylsl`
falla, instálala (por ejemplo con Homebrew: `brew install labstreaminglayer/tap/lsl`)
o apunta `PYLSL_LIB` al `.dylib`/`.so`.

## Uso

### Listar streams

```bash
uv run pipeline list
uv run pipeline list --type EEG --wait 3
```

### Leer muestras

```bash
uv run pipeline read --type EEG
uv run pipeline read --name OpenBCI --duration 5
uv run pipeline read --type EEG --max-chunks 10
```

### Visualizador en vivo

Abre una ventana con las formas de onda multi-canal que llegan por LSL.
Si el simulador publica un stream `*_Markers`, cada evento abre un tramo sombreado
hasta el siguiente: rest (T0), left (T1) y right (T2) se ven completos, con una
línea y su etiqueta en el inicio.

```bash
uv run pipeline view --type EEG
uv run pipeline view --name OpenBCI --window 8
uv run pipeline view --type EEG --channels 0-7 --refresh 30
uv run pipeline view --name OpenBCI_EEG
```

Opciones útiles:

| Flag | Descripción |
|------|-------------|
| `--window` | Segundos visibles en pantalla (default: 5) |
| `--channels` | Índices a dibujar, p.ej. `0,1,2` o `0-7` (default: todos) |
| `--scale` | Separación vertical fija entre canales (default: auto) |
| `--refresh` | Hz de redibujo de la UI (default: 30) |
| `--markers-name` | Nombre del stream LSL Markers (default: `<eeg>_Markers`) |
| `--no-markers` | No buscar stream de marcadores |

### API Python

```python
from pipeline import LslStreamReader, list_streams

for stream in list_streams(stream_type="EEG", wait_seconds=2.0):
    print(stream.display_name)

with LslStreamReader.open(stream_type="EEG", wait_seconds=5.0) as reader:
    chunk = reader.pull_chunk(timeout=1.0, max_samples=64)
    print(len(chunk.samples), chunk.timestamps[:3])
```

## Flujo con el simulador

```bash
# Terminal 1
cd ../simulator && uv run simopenbci

# Terminal 2
cd ../pipeline
uv run pipeline list
uv run pipeline view --type EEG
```

## Tests

```bash
uv run pytest -q
```
