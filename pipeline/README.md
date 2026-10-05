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

### Grabar una sesión Cyton

El servicio local es el que obedece al botón Start del game. La Cyton entra por LSL: 8 canales, 250 Hz, µV. BIAS (ground) y SRB2 (referencia) se guardan como metadatos, no como columnas.

```bash
uv run pipeline serve
uv run pipeline record --session-id demo --duration 10
uv run pipeline show recordings/demo
```

Cada sesión es una carpeta:

| Archivo | Contenido |
|---------|-----------|
| `signal.f32` | float32 little-endian, `(muestras, 8)`, µV |
| `time.f64` | timestamp LSL por muestra EEG |
| `emg.f32` | float32 little-endian, `(muestras EMG, 8)`, antebrazo |
| `emg_time.f64` | timestamp LSL por muestra EMG |
| `markers.jsonl` | kind, code, label y tiempo LSL |
| `session.json` | perfil de la Cyton, montaje EMG y etiquetas |

Offline y Online real leen además un stream LSL de tipo `EMG` con 8 electrodos: 4 en el antebrazo izquierdo y 4 en el derecho. En cada antebrazo, 2 van en la cara anterior y 2 en la posterior. El orden de `emg.f32` es anterior izquierdo, posterior izquierdo, anterior derecho, posterior derecho. Ese archivo se guarda en crudo. El acondicionamiento, aparte, es pasa-banda 10–250 Hz, notch a la frecuencia de red (50 Hz), rectificación de onda completa y luego la envolvente: RMS en ventanas de 100 ms, o la media de la señal rectificada en esa misma ventana.

El simulador no escribe esa carpeta. La lee con `--dataset recorded_session --file <carpeta>`.

### API del servicio

`http://127.0.0.1:8765`

| Método | Ruta | Uso |
|--------|------|-----|
| POST | `/v1/record/start` | Empieza a grabar el EEG Cyton y el EMG de antebrazo (`session_id`) |
| POST | `/v1/record/marker` | Anota un trial (`kind`, `code`, `label`) |
| POST | `/v1/record/stop` | Cierra la carpeta |
| GET | `/v1/datasets` | Datasets que el simulador puede reproducir |
| GET | `/v1/subjects?dataset=` | Participantes de ese dataset |
| POST | `/v1/online/start` | Arranca el simulador o escucha la Cyton |
| POST | `/v1/online/marker` | Online real: la fase de actividad del trial (`imagery_start`, `imagery_end`) |
| POST | `/v1/online/stop` | Detiene la corrida online |
| POST | `/v1/pre-model/filter` | Primer módulo offline: notch y luego pasa-banda de un participante |
| GET | `/v1/signal` | Últimos segundos de EEG de una corrida en vivo |
| GET | `/v1/status` | Muestras grabadas o vistas |

Pre model construye el modelo offline. El primer módulo filtra la señal continua antes de epoching: notch y después un pasa-banda. La banda por defecto es 8–30 Hz, la de los ritmos sensorimotores para imaginación de mano izquierda/derecha. `POST /v1/pre-model/filter` devuelve esa ventana en crudo y ya filtrada. Características y clasificador siguen pendientes.

Online real no busca la actividad en LSL: la sesión ya define cada trial, y en esa fase el pipeline corta la señal. Online simulated toma el inicio de la actividad del dataset (Liu: marcador de imaginación; Shuqfa: el periodo de la tarea; una sesión grabada: `trial_start` de la mano). En ambos casos el ventaneo es el mismo: cinco ventanas de 2 s con paso de 0,5 s, de 0–2 a 2–4, medidas desde el inicio de la actividad.

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
