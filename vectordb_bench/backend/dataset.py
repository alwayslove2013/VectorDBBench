"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

from collections import namedtuple
import logging
import pathlib
from enum import Enum
import pandas as pd
from pydantic import validator, PrivateAttr
import polars as pl
from pyarrow.parquet import ParquetFile

from vectordb_bench.backend.filters import Filter, LabelFilter, MSMarcoV2LabelFilter

from ..base import BaseModel
from .. import config
from ..backend.clients import MetricType
from . import utils
from .data_source import DatasetSource, DatasetReader

log = logging.getLogger(__name__)


SizeLabel = namedtuple("SizeLabel", ["size", "label", "file_count"])


class BaseDataset(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    with_gt: bool = False
    _size_label: dict[int, SizeLabel] = PrivateAttr()
    is_custom: bool = False
    with_remote_resource: bool = True
    with_scalar_labels: bool = False
    scalar_labels_file_separated: bool = True
    scalar_labels_file: str = "scalar_labels.parquet"
    scalar_label_percentages: list[float] = []

    train_id_field: str = "id"
    train_vector_field: str = "emb"
    test_id_field: str = "id"
    test_vector_field: str = "emb"
    gt_id_field: str = "id"
    gt_neighbors_field: str = "neighbors_id"

    @validator("size")
    def verify_size(cls, v):
        if v not in cls._size_label:
            raise ValueError(
                f"Size {v} not supported for the dataset, expected: {cls._size_label.keys()}"
            )
        return v

    @property
    def label(self) -> str:
        return self._size_label.get(self.size).label

    @property
    def full_name(self) -> str:
        return f"{self.name.capitalize()} ({self.label.capitalize()})"

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()

    @property
    def file_count(self) -> int:
        return self._size_label.get(self.size).file_count

    @property
    def train_files(self) -> list[str]:
        return utils.compose_train_files(self.file_count, self.use_shuffled)


class CustomDataset(BaseDataset):
    dir: str
    file_num: int
    is_custom: bool = True
    with_remote_resource: bool = False

    @validator("size")
    def verify_size(cls, v):
        return v

    @property
    def label(self) -> str:
        return "Custom"

    @property
    def dir_name(self) -> str:
        return self.dir

    @property
    def file_count(self) -> int:
        return self.file_num


class LAION(BaseDataset):
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    with_gt: bool = True
    _size_label: dict = {
        100_000_000: SizeLabel(100_000_000, "LARGE", 100),
    }


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = (True,)
    _size_label: dict = {
        100_000: SizeLabel(100_000, "SMALL", 1),
        1_000_000: SizeLabel(1_000_000, "MEDIUM", 1),
        10_000_000: SizeLabel(10_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
    ]


class Glove(BaseDataset):
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    _size_label: dict = {1_000_000: SizeLabel(1_000_000, "MEDIUM", 1)}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        500_000: SizeLabel(
            500_000,
            "SMALL",
            1,
        ),
        5_000_000: SizeLabel(5_000_000, "MEDIUM", 1),
        #  50_000_000: SizeLabel(50_000_000, "LARGE", 50),
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    with_gt: bool = (True,)
    _size_label: dict = {
        50_000: SizeLabel(50_000, "SMALL", 1),
        500_000: SizeLabel(500_000, "MEDIUM", 1),
        5_000_000: SizeLabel(5_000_000, "LARGE", 10),
    }
    with_scalar_labels: bool = True
    scalar_label_percentages: list[float] = [
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
    ]


class MSMarcoV2(BaseDataset):
    name: str = "MSMarcoV2"
    size: int = 138_000_000
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    with_gt: bool = True
    is_custom: bool = True
    with_remote_resource: bool = False  # todo
    with_scalar_labels: bool = True
    scalar_labels_file_separated: bool = False
    scalar_label_percentages: list[float] = [0.01, 0.05, 0.1, 0.2, 0.4]
    train_id_field: str = "pk"
    train_vector_field: str = "float32_vector"

    @validator("size")
    def verify_size(cls, v):
        return v

    @property
    def label(self) -> str:
        return "138M"

    @property
    def dir_name(self) -> str:
        return "msmarco_v2_138M_parquet"

    @property
    def file_count(self) -> int:
        return 69 * 4

    @property
    def train_files(self) -> list[str]:
        return ["train.parquet"]  # todo
        return [
            f"msmarco_passage_{i:02}-{j}.parquet" for i in range(69) for j in range(4)
        ]


class DatasetManager(BaseModel):
    """Download dataset if not in the local directory. Provide data for cases.

    DatasetManager is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere = Dataset.COHERE.manager(100_000)
        >>> for data in cohere:
        >>>    print(data.columns)
    """

    data: BaseDataset
    test_data: list[list[float]] | None = None
    gt_data: list[list[int]] | None = None
    scalar_labels: pd.DataFrame | None = None
    train_files: list[str] = []
    reader: DatasetReader | None = None

    def __eq__(self, obj):
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    def set_reader(self, reader: DatasetReader):
        self.reader = reader

    @property
    def data_dir(self) -> pathlib.Path:
        """data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.relative_path
            '/tmp/vectordb_bench/dataset/sift/sift_small_500k/'
        """
        # return pathlib.Path(
        #     config.DATASET_LOCAL_DIR, self.data.name.lower(), self.data.dir_name.lower()
        # )
        return pathlib.Path(config.DATASET_LOCAL_DIR, self.data.dir_name)

    def __iter__(self):
        return DataSetIterator(self)

    # TODO passing use_shuffle from outside
    def prepare(
        self,
        filters: Filter,
        source: DatasetSource = DatasetSource.S3,
    ) -> bool:
        """Download the dataset from DatasetSource
         url = f"{source}/{self.data.dir_name}"

        Args:
            source(DatasetSource): S3 or AliyunOSS, default as S3
            filters(Optional[int | float | str]): combined with dataset's with_gt to
              compose the correct ground_truth file

        Returns:
            bool: whether the dataset is successfully prepared

        """
        self.train_files = self.data.train_files
        scalar_labels_file = None
        gt_file, test_file = None, None
        if self.data.with_gt:
            gt_file, test_file = filters.get_groundtruth_file(), "test.parquet"

        if self.data.with_remote_resource:
            download_files = [file for file in self.train_files]
            if self.data.with_scalar_labels and self.data.scalar_labels_file_separated:
                download_files.append(self.data.scalar_labels_file)
            download_files.extend([gt_file, test_file])
            source.reader().read(
                dataset=self.data.dir_name.lower(),
                files=download_files,
                local_ds_root=self.data_dir,
            )

        if self.data.with_scalar_labels:
            if self.data.scalar_labels_file_separated:
                self.scalar_labels = self._read_file(scalar_labels_file)
            else:
                # read with train_file
                pass

        if gt_file is not None and test_file is not None:
            self.test_data = self._read_file(test_file)[
                self.data.test_vector_field
            ].to_list()
            self.gt_data = self._read_file(gt_file)[
                self.data.gt_neighbors_field
            ].to_list()

        # prefix = "shuffle_train" if use_shuffled else "train"
        # self.train_files = sorted(
        #     [f.name for f in self.data_dir.glob(f"{prefix}*.parquet")]
        # )
        log.debug(f"{self.data.name}: available train files {self.train_files}")

        return True

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = pathlib.Path(self.data_dir, file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pd.DataFrame()

        return pl.read_parquet(p)


class DataSetIterator:
    def __init__(self, dataset: DatasetManager):
        self._ds = dataset
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [
            0 for i in range(len(self._ds.train_files))
        ]  # iter num for each file

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            raise IndexError(f"No such file {p}")
            log.warning(f"No such file: {p}")
        return ParquetFile(p).iter_batches(config.NUM_PER_BATCH)

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            if self._cur is None:
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)

            try:
                return next(self._cur).to_pandas()
            except StopIteration:
                if self._idx == len(self._ds.train_files) - 1:
                    raise StopIteration from None

                self._idx += 1
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)
                return next(self._cur).to_pandas()
        raise StopIteration


class Dataset(Enum):
    """
    Value is Dataset classes, DO NOT use it
    Example:
        >>> all_dataset = [ds.name for ds in Dataset]
        >>> Dataset.COHERE.manager(100_000)
        >>> Dataset.COHERE.get(100_000)
    """

    LAION = LAION
    GIST = GIST
    COHERE = Cohere
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI
    MSMarcoV2 = MSMarcoV2

    def get(self, size: int) -> BaseDataset:
        return self.value(size=size)

    def manager(self, size: int) -> DatasetManager:
        return DatasetManager(data=self.get(size))


class DatasetWithSizeType(Enum):
    CohereMedium = "Medium Cohere (768dim, 1M)"
    CohereLarge = "Large Cohere (768dim, 10M)"
    OpenAIMedium = "Medium OpenAI (1536dim, 500K)"
    OpenAILarge = "Large OpenAI (1536dim, 5M)"
    MSMarcoV2 = "MS Marco V2 (1536dim, 138M)"

    def get_manager(self) -> DatasetManager:
        if self not in DatasetWithSizeMap:
            raise ValueError(f"wrong ScalarDatasetWithSizeType: {self.name}")
        return DatasetWithSizeMap.get(self)

    def get_load_timeout(self) -> float:
        if "medium" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.LOAD_TIMEOUT_768D_10M
        if "msmarco" in self.value.lower().replace(" ", ""):
            return config.LOAD_TIMEOUT_1536_138M
        raise KeyError(f"No load_timeout for {self.value}")

    def get_optimize_timeout(self) -> float:
        if "medium" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_1M
        if "large" in self.value.lower():
            return config.OPTIMIZE_TIMEOUT_768D_10M
        if "msmarco" in self.value.lower().replace(" ", ""):
            return config.OPTIMIZE_TIMEOUT_1536_138M
        return config.OPTIMIZE_TIMEOUT_DEFAULT

    @property
    def labels_filter_cls(self) -> type[LabelFilter]:
        if "msmarco" in self.value.lower().replace(" ", ""):
            return MSMarcoV2LabelFilter
        return LabelFilter


DatasetWithSizeMap = {
    DatasetWithSizeType.CohereMedium: Dataset.COHERE.manager(1_000_000),
    DatasetWithSizeType.CohereLarge: Dataset.COHERE.manager(10_000_000),
    DatasetWithSizeType.OpenAIMedium: Dataset.OPENAI.manager(500_000),
    DatasetWithSizeType.OpenAILarge: Dataset.OPENAI.manager(5_000_000),
    DatasetWithSizeType.MSMarcoV2: Dataset.MSMarcoV2.manager(138_000_000),
}
