"""
Loader genérico para archivos CSV/TXT con formato legacy del simulador.
Líneas que empiezan con '%' se ignoran; columnas configurables para canales EEG.
"""
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, DatasetMetadata
from .parsing import iter_float_rows


class GenericCSVLoader(BaseDatasetLoader):
    """
    Formato: una línea por muestra, valores separados por coma.
    Líneas que empiezan con '%' son comentarios.
    Por defecto: columnas 1..8 (índices 1:9) = 8 canales (compatible con SimOpenBCI original).
    """

    id = "generic_csv"
    label = "Generic CSV"

    def __init__(
        self,
        n_channels: int = 8,
        channel_start_index: int = 1,
        sampling_rate: float = 250.0,
        stream_name: str = "BioSemi",
        comment_prefix: str = "%",
    ) -> None:
        self._channel_start = channel_start_index
        self._comment_prefix = comment_prefix
        self._metadata = DatasetMetadata(
            name=stream_name,
            n_channels=n_channels,
            sampling_rate=sampling_rate,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    def iter_samples(self, file_path: str | Path) -> Iterator[list[float]]:
        yield from iter_float_rows(
            file_path,
            n_channels=self._metadata.n_channels,
            start_col=self._channel_start,
            comment_prefix=self._comment_prefix,
        )

    def can_handle(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in (".txt", ".csv")
