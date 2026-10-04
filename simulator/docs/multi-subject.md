# Datasets con múltiples sujetos

Para que un dataset funcione con **todos los sujetos** desde línea de comandos (`--list-subjects`, `--subject`, `--run`), el loader debe implementar dos métodos de clase en la **base**:

## 1. `list_subjects(data_root)` → `list[SubjectRunInfo]`

- **Entrada**: `data_root`, carpeta raíz del dataset (p. ej. `data/sujetos_BCI_Competition_IV2a`).
- **Salida**: Lista de `SubjectRunInfo(subject_id="A01", runs=["T", "E"])` con todos los sujetos y los runs/sesiones disponibles.
- **Uso**: Se llama con `--list-subjects` para mostrar la tabla de sujetos y runs.

## 2. `resolve_subject_file(data_root, subject_id, run=None)` → `Path | None`

- **Entrada**: misma `data_root`, `subject_id` (ej. `"A03"`), y opcionalmente `run` (ej. `"T"`).
- **Salida**: Ruta del archivo a streamear para ese sujeto y run, o `None` si no existe.
- **Uso**: Con `--subject A03 --run T` el simulador usa esta ruta en lugar de `--file`.

## 3. `default_data_root()` → `Path | None` (opcional)

- Devuelve la carpeta por defecto bajo el cwd (ej. `data/sujetos_BCI_Competition_IV2a`).
- Si se implementa, `--list-subjects` / `--subject` no requieren `--data-root`.

## Convención de nombres (ejemplo BCI IV 2a)

- Archivos: `{subject_id}{run}.txt` → `A01T.txt`, `A01E.txt`.
- Para otro dataset puede usarse cualquier esquema (p. ej. `sujeto_01_sesion_entrenamiento.csv`); solo hay que implementar la lógica en `list_subjects` y `resolve_subject_file` de forma coherente.

## Ejemplo: añadir un dataset “OpenBMI” con sujetos

1. Crear `openbmi_loader.py` con una clase que herede de `BaseDatasetLoader`.
2. Implementar `iter_samples`, `metadata`, y si aplica `detect_from_path` / `default_data_root`.
3. Para multi-sujeto, implementar:
   - `list_subjects(data_root)`: escanear `data_root` y devolver lista de sujetos y sus runs.
   - `resolve_subject_file(data_root, subject_id, run)`: devolver `Path` al archivo correspondiente.
4. Registrar el loader en `__init__.py` en `LOADERS`.

Así `uv run simopenbci --dataset openbmi --list-subjects` y `--subject X --run Y` funcionarán **sin cambiar** `simopenbci/cli.py`.
