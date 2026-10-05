# Liu et al. 2024 – imaginación de agarre en ACV

Dataset de [Liu et al., Scientific Data 2024](https://doi.org/10.1038/s41597-023-02787-8) (PMC10811218): EEG de imaginación motora de agarre en 50 pacientes con ACV agudo. DOI de los archivos: [10.6084/m9.figshare.21393402](https://doi.org/10.6084/m9.figshare.21393402).

El simulador reproduce el EDF preprocesado (`edffile/`), no el `.mat` crudo de `sourcedata/`.

## Paradigma

El participante está sentado frente a una pantalla (~80 cm), con instrucciones por audio. Cada ensayo dura 8 s y se alternan mano izquierda y derecha (20 + 20). Un run tiene 40 ensayos (~320 s de registro de tarea).

| Etapa | Duración | Actividad | Marcador |
|-------|----------|-----------|----------|
| Instrucción | 2 s | Imagen y texto de la mano a imaginar | 1 |
| Imaginación motora | 4 s | Imaginar agarrar un objeto esférico, con video del movimiento | 2 |
| Descanso | 2 s | Relajarse | 3 |

La clase publicada es la mano: `1` = izquierda, `2` = derecha. El reposo no es una clase. En los 50 sujetos el orden es el mismo: izquierda, derecha, y así hasta 40 ensayos.

El artículo describe además un minuto con ojos abiertos y otro con ojos cerrados antes de la tarea. Esos bloques no están en el EDF: cada archivo dura 320 s, exactamente los 40 ensayos.

## Adquisición

- Amplificador ZhenTec NT1, 500 Hz, red de 50 Hz.
- Referencia CPz, tierra FPz, montaje 10-10.
- 29 EEG de registro. El EDF también trae CPz (referencia, constante 0), HEOL, HEOR y un canal de marcadores sin nombre; el simulador no los emite.
- El EDF está en µV, con baseline y filtrado 0.5–40 Hz hechos por los autores.

`task-motor-imagery_channels.tsv` lista 31 nombres y llama VEOR al segundo EOG. El EDF trae 33 señales y el segundo EOG se llama HEOR.

## Marcadores

El canal sin nombre guarda un impulso de una muestra, no un nivel sostenido:

- `1` inicio de instrucción
- `2` inicio de imaginación (el instante que el código de los autores usa como trigger)
- `3` inicio de descanso

`task-motor-imagery_events.tsv` es el horario nominal (onset en ms). En el EDF la instrucción cae en la rejilla de 8 s, pero imaginación y descanso se desplazan unos pocos milisegundos. El simulador usa los impulsos del archivo.

Esos impulsos salen por el stream LSL de marcadores:

| kind | code | label |
|------|------|-------|
| `instruction` | 1 | `left` o `right` |
| `trial_start` | 2 | `left` o `right` |
| `rest` | 3 | `rest` |

## Uso

Extraer el zip de Figshare y, dentro, `edffile.zip`:

```bash
mkdir -p data/liu2024_stroke
unzip -d data/liu2024_stroke /ruta/21679035.zip
unzip -d data/liu2024_stroke data/liu2024_stroke/edffile.zip

uv run simopenbci --dataset liu2024_stroke --list-subjects
uv run simopenbci --dataset liu2024_stroke --subject sub-01 --no-dialog
```

`--subject 1` y `--subject sub-01` resuelven el mismo archivo. El stream lleva los 29 EEG a 500 Hz, nombre `Liu2024_Stroke_MI`. `--channels 8` toma los ocho primeros (FP1…FCz).
