# BCI Competition IV Dataset 2a – Referencia

Esta es la descripción oficial del dataset que se transmite desde el simulador (archivos TXT derivados del formato GDF original). Sirve como referencia para saber qué datos se están enviando por LSL.

---

## Descripción del dataset

The data set consists of **EEG data from 9 subjects**. The cue-based BCI paradigm consisted of **four different motor imagery tasks**:

- **Class 1**: Imagination of movement of the **left hand**
- **Class 2**: **Right hand**
- **Class 3**: **Both feet**
- **Class 4**: **Tongue**

**Sessions**: Two sessions on different days were recorded for each subject. Each session is comprised of **6 runs** separated by short breaks. One run consists of **48 trials** (12 for each of the four possible classes), yielding a total of **288 trials per session**.

At the beginning of each session, a recording of approximately **5 minutes** was performed to estimate the EOG influence. The recording was divided into 3 blocks: (1) two minutes with eyes open (looking at a fixation cross on the screen), (2) one minute with eyes closed, and (3) one minute with eye movements. Note: due to technical problems the EOG block is shorter for subject **A04T** and contains only the eye movement condition.

**Paradigm**: The subjects were sitting in a comfortable armchair in front of a computer screen. At the beginning of a trial (t = 0 s), a fixation cross appeared on the black screen. In addition, a short acoustic warning tone was presented. After **two seconds (t = 2 s)**, a **cue** in the form of an arrow (left, right, down or up) appeared and stayed on the screen for **1.25 s**. This prompted the subjects to perform the desired motor imagery task. No feedback was provided. The subjects were asked to carry out the motor imagery task until the fixation cross disappeared from the screen at **t = 6 s**.

---

## Adquisición

- **EEG**: **22 Ag/AgCl electrodes** (inter-electrode distance 3.5 cm), montage according to international 10–20 system. Monopolar recording with **left mastoid reference**, right mastoid ground.
- **Sampling**: **250 Hz**.
- **Filtering**: Bandpass **0.5–100 Hz**, **50 Hz notch** enabled. Amplifier sensitivity **100 µV**.
- **EOG**: **3 monopolar EOG channels** (250 Hz, 0.5–100 Hz, 50 Hz notch, 1 mV sensitivity). Provided for artifact processing; **must not be used for classification**.

In our simulator, the **TXT files contain 22 EEG channels** (columns 1–22; column 0 = time). EOG can be included in the original GDF but the current TXT export uses 22 channels.

---

## Archivos (tabla del dataset)

| ID | Training | Evaluation |
|----|----------|------------|
| 1  | A01T.gdf | A01E.gdf   |
| 2  | A02T.gdf | A02E.gdf   |
| 3  | A03T.gdf | A03E.gdf   |
| 4  | A04T.gdf | A04E.gdf   |
| 5  | A05T.gdf | A05E.gdf   |
| 6  | A06T.gdf | A06E.gdf   |
| 7  | A07T.gdf | A07E.gdf   |
| 8  | A08T.gdf | A08E.gdf   |
| 9  | A09T.gdf | A09E.gdf   |

In this simulator, the corresponding **TXT** files are named `A01T.txt`, `A01E.txt`, … `A09T.txt`, `A09E.txt`.

---

## Event types (GDF / BioSig)

For reference (relevant when loading original GDF or building epochs in articulo_1):

| Decimal | Hex     | Description              |
|---------|--------|---------------------------|
| 276     | 0x0114 | Idling EEG (eyes open)   |
| 277     | 0x0115 | Idling EEG (eyes closed) |
| 768     | 0x0300 | Start of a trial         |
| **769** | 0x0301 | **Cue onset left (class 1)**  |
| **770** | 0x0302 | **Cue onset right (class 2)** |
| **771** | 0x0303 | **Cue onset foot (class 3)**  |
| **772** | 0x0304 | **Cue onset tongue (class 4)**|
| 783     | 0x030F | Cue unknown              |
| **1023**| 0x03FF | **Rejected trial** (artifact)|
| 1072    | 0x0430 | Eye movements            |
| 32766   | 0x7FFE | Start of a new run       |

Runs are separated by 100 missing values (NaN in BioSig). Class labels (1–4) are only provided for the **training** data; evaluation data has no labels. Trials marked as artifacts are event type 1023; `h.ArtifactSelection` in the header lists clean (0) vs artifact (1) per trial.

---

## Uso en el simulador

- **Streaming**: El simulador envía las muestras en tiempo real (o según `--max-samples`). No envía eventos; solo canal de EEG (22 o 8 con `--channels 8`).
- **Sujetos y runs**: `--list-subjects`, `--subject A01` … `A09`, `--run T` o `--run E`.

Para **entrenamiento/evaluación con etiquetas y eventos**, usar el proyecto **articulo_1** y el loader `bci_4class_22ch` con los GDF o con datos exportados (p. ej. numpy) que incluyan eventos; ver la documentación del dataset en articulo_1.
