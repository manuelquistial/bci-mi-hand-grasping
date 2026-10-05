# Datasets con múltiples sujetos

Para que un dataset funcione con **todos los sujetos** desde línea de comandos (`--list-subjects`, `--subject`, `--run`), el loader debe implementar dos métodos de clase en la **base**:

## 1. `list_subjects(data_root)` → `list[SubjectRunInfo]`

- **Entrada**: `data_root`, carpeta raíz del dataset (p. ej. `data/shuqfa2024`).
- **Salida**: Lista de `SubjectRunInfo(subject_id="S001", runs=["01", "02"])` con todos los sujetos y los runs/sesiones disponibles.
- **Uso**: Se llama con `--list-subjects` para mostrar la tabla de sujetos y runs.

## 2. `resolve_subject_file(data_root, subject_id, run=None)` → `Path | None`

- **Entrada**: misma `data_root`, `subject_id` (ej. `"S001"`), y opcionalmente `run` (ej. `"02"`).
- **Salida**: Ruta del archivo a streamear para ese sujeto y run, o `None` si no existe.
- **Uso**: Con `--subject S001 --run 02` el simulador usa esta ruta en lugar de `--file`.

## 3. `default_data_root()` → `Path | None` (opcional)

- Devuelve `data/<id>/` bajo el cwd (ej. `data/shuqfa2024`).
- Si se implementa, `--list-subjects` / `--subject` no requieren `--data-root`.

## Convención de nombres

- Cada dataset elige su esquema. Shuqfa 2024 usa `SUB_001_SIG_02.csv`.
- Solo hay que implementar `list_subjects` y `resolve_subject_file` de forma coherente con esos nombres.

## Ejemplo: añadir un dataset “OpenBMI” con sujetos

1. Crear `openbmi_loader.py` con una clase que herede de `BaseDatasetLoader`.
2. Implementar `iter_samples`, `metadata`, y si aplica `detect_from_path` / `default_data_root`.
3. Para multi-sujeto, implementar:
   - `list_subjects(data_root)`: escanear `data_root` y devolver lista de sujetos y sus runs.
   - `resolve_subject_file(data_root, subject_id, run)`: devolver `Path` al archivo correspondiente.
4. Registrar el loader en `__init__.py` en `LOADERS`.

Así `uv run simopenbci --dataset openbmi --list-subjects` y `--subject X --run Y` funcionarán **sin cambiar** `simopenbci/cli.py`.
