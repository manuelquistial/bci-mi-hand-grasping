# Shuqfa et al. 2024 – PhysioNet EEGBCI depurado

Versión limpia del [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) publicada por [Shuqfa et al., Data in Brief 2024](https://doi.org/10.1016/j.dib.2024.110181). Archivos: [Mendeley 10.17632/dpmtgrn8d8.4](https://doi.org/10.17632/dpmtgrn8d8.4).

Es el EEG Motor Movement/Imagery de PhysioNet. El simulador lee los CSV (`SUB_[sujeto]_SIG_[registro].csv`).

## Qué se transmite

103 sujetos. Se quitaron las dos líneas base (ojos abiertos y ojos cerrados) y seis sujetos con registros inconsistentes. Quedan 12 registros de tarea por sujeto, cada uno en su propio CSV.

Cada registro se envía solo. `--run 2` y `--run 02` apuntan al mismo archivo. Sin `--run`, el simulador envía el `02`.

| Registros depurados | Original | Tarea | Etiquetas |
|---------------------|----------|-------|-----------|
| 01, 05, 09 | 3, 7, 11 | Ejecución, un puño | 1 reposo, 2 izquierda, 3 derecha |
| 02, 06, 10 | 4, 8, 12 | Imaginación, un puño | 4 reposo, 5 izquierda, 6 derecha |
| 03, 07, 11 | 5, 9, 13 | Ejecución, ambos puños o pies | 7 reposo, 8 ambos puños, 9 ambos pies |
| 04, 08, 12 | 6, 10, 14 | Imaginación, ambos puños o pies | 10 reposo, 11 ambos puños, 12 ambos pies |

## Señal y marcadores

- 64 canales, 160 Hz, orden 10-10 de EEGBCI (Fc5 … Iz).
- El CSV de anotaciones del mismo registro (`SUB_001_ANN_02.csv`) tiene una fila por periodo: etiqueta, duración en segundos, duración en muestras, índice de inicio e índice de fin. El inicio está en base 1.
- La duración real sale de esa fila. El simulador pone el marcador en la muestra de inicio.

En el stream de marcadores, reposo (`1`, `4`, `7`, `10`) sale como `rest`. El resto sale como `trial_start`, con `left`, `right`, `both_fists` o `both_feet`.

## Uso

El zip trae `eegmmidb.rar` (WinRAR) con los 2.472 CSV. Hace falta un descompresor RAR (`unar` o `unrar`):

```bash
mkdir -p data/shuqfa2024
unzip -j -d data/shuqfa2024 "/ruta/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation.zip" "*eegmmidb.rar"
unar -o data/shuqfa2024 data/shuqfa2024/eegmmidb.rar

uv run simopenbci --dataset shuqfa2024 --list-subjects
uv run simopenbci --dataset shuqfa2024 --subject S001 --run 2 --no-dialog
uv run simopenbci --dataset shuqfa2024 --subject 1 --run 6 --no-dialog
uv run simopenbci --dataset shuqfa2024 --subject 1 --run 10 --no-dialog
```

`--subject 1` y `--subject S001` resuelven `SUB_001_SIG_02.csv`. El stream se llama `Shuqfa2024_EEG`. `--channels 8` toma Fc5 … Fc6.
