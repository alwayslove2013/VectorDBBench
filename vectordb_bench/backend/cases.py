import logging
from enum import Enum, auto
from typing import Type

from vectordb_bench import config
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.filters import (
    Filter,
    FilterType,
    IntFilter,
    NonFilter,
)
from vectordb_bench.base import BaseModel
from vectordb_bench.frontend.components.custom.getCustomConfig import (
    CustomDatasetConfig,
)
from .dataset import CustomDataset, Dataset, DatasetManager, DatasetWithSizeType


log = logging.getLogger(__name__)


class CaseType(Enum):
    """
    Example:
        >>> case_cls = CaseType.CapacityDim128.case_cls
        >>> assert c is not None
        >>> CaseType.CapacityDim128.case_name
        "Capacity Test (128 Dim Repeated)"
    """

    CapacityDim128 = 1
    CapacityDim960 = 2

    Performance768D100M = 3
    Performance768D10M = 4
    Performance768D1M = 5

    Performance768D10M1P = 6
    Performance768D1M1P = 7
    Performance768D10M99P = 8
    Performance768D1M99P = 9

    Performance1536D500K = 10
    Performance1536D5M = 11

    Performance1536D500K1P = 12
    Performance1536D5M1P = 13
    Performance1536D500K99P = 14
    Performance1536D5M99P = 15

    Performance1536D50K = 50

    Custom = 100
    PerformanceCustomDataset = 101
    LabelFilterPerformanceCase = 102
    PerformanceMSMarcoV2 = 60

    def case_cls(self, custom_configs: dict | None = None) -> Type["Case"]:
        if custom_configs is None:
            return type2case.get(self)()
        else:
            return type2case.get(self)(**custom_configs)

    def case_name(self, custom_configs: dict | None = None) -> str:
        c = self.case_cls(custom_configs)
        if c is not None:
            return c.name
        raise ValueError("Case unsupported")

    def case_description(self, custom_configs: dict | None = None) -> str:
        c = self.case_cls(custom_configs)
        if c is not None:
            return c.description
        raise ValueError("Case unsupported")


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """Undefined case

    Fields:
        case_id(CaseType): default 9 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | None): one of 99% | 1% | None
        filters(Filter): NonFilter / IntFilter / LabelFilter
        with_scalar_labels(bool): whether to insert scalar data (labels)
    """

    case_id: CaseType
    label: CaseLabel
    name: str
    description: str
    dataset: DatasetManager
    with_scalar_labels: bool = False

    load_timeout: float | int
    optimize_timeout: float | int | None = None

    filter_rate: float | None = None
    filters: Filter = NonFilter()

    def check_scalar_labels(self) -> None:
        if self.with_scalar_labels and not self.dataset.data.with_scalar_labels:
            raise ValueError(
                f"""no scalar_labels data in current dataset ({self.dataset.data.name})"""
            )
        if self.filters.type == FilterType.Label and not self.with_scalar_labels:
            raise ValueError("label-filter cases need scalar_labels data")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # post check
        self.check_scalar_labels()

    # @property
    # def filters(self) -> dict | None:
    #     if self.filter_rate is not None:
    #         ID = round(self.filter_rate * self.dataset.data.size)
    #         return {
    #             "metadata": f">={ID}",
    #             "id": ID,
    #         }

    #     return None


class CapacityCase(Case):
    label: CaseLabel = CaseLabel.Load
    filter_rate: float | None = None
    load_timeout: float | int = config.CAPACITY_TIMEOUT_IN_SECONDS
    optimize_timeout: float | int | None = None


class PerformanceCase(Case):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | None = None
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT


class IntFilterPerformanceCase(PerformanceCase):
    @property
    def filters(self) -> Filter:
        int_field = "id"
        int_value = self.dataset.data.size * self.filter_rate
        return IntFilter(
            filter_rate=self.filter_rate, int_field=int_field, int_value=int_value
        )


class CapacityDim960(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim960
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension 
     vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded. Number of inserted vectors will be 
     reported."""


class CapacityDim128(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim128
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension
     vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded. Number of inserted vectors will be 
     reported."""


class Performance768D10M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M1P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M1P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M99P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M99P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D100M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.LAION.manager(100_000_000)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_100M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_100M


class Performance1536D500K(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Search Performance Test (500K Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 500K dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Search Performance Test (5M Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 5M dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D500K1P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M1P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D500K99P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M99P(IntFilterPerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D50K(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D50K
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(50_000)
    name: str = "Search Performance Test (50K Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 50K dataset (<b>OpenAI 50K vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = 3600
    optimize_timeout: float | int | None = 15 * 60


def metric_type_map(s: str) -> MetricType:
    if s.lower() == "cosine":
        return MetricType.COSINE
    if s.lower() == "l2" or s.lower() == "euclidean":
        return MetricType.L2
    if s.lower() == "ip":
        return MetricType.IP
    err_msg = f"Not support metric_type: {s}"
    log.error(err_msg)
    raise RuntimeError(err_msg)


class LabelFilterPerformanceCase(PerformanceCase):
    case_id: CaseType = CaseType.LabelFilterPerformanceCase
    with_scalar_labels: bool = True
    dataset_with_size_type: DatasetWithSizeType

    def __init__(
        self,
        dataset_with_size_type: DatasetWithSizeType,
        label_percentage: float,
        **kwargs,
    ):
        if not isinstance(dataset_with_size_type, DatasetWithSizeType):
            dataset_with_size_type = DatasetWithSizeType(dataset_with_size_type)
        name = (
            f"Label-Filter-{label_percentage*100:.1f}% - {dataset_with_size_type.value}"
        )
        description = f"Label-Filter-{label_percentage*100:.1f}% Performance Test ({dataset_with_size_type.value})"
        dataset = dataset_with_size_type.get_manager()
        load_timeout = dataset_with_size_type.get_load_timeout()
        optimize_timeout = dataset_with_size_type.get_optimize_timeout()
        filter = dataset_with_size_type.labels_filter_cls(
            label_percentage=label_percentage
        )
        filter_rate = filter.filter_rate
        super().__init__(
            name=name,
            description=description,
            dataset=dataset,
            load_timeout=load_timeout,
            optimize_timeout=optimize_timeout,
            filter_rate=filter_rate,
            filters=filter,
            dataset_with_size_type=dataset_with_size_type,
            **kwargs,
        )


class PerformanceCustomDataset(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceCustomDataset
    name: str = "Performance With Custom Dataset"
    description: str = ""
    dataset: DatasetManager

    def __init__(
        self,
        name,
        description,
        load_timeout,
        optimize_timeout,
        dataset_config,
        **kwargs,
    ):
        dataset_config = CustomDatasetConfig(**dataset_config)
        dataset = CustomDataset(
            name=dataset_config.name,
            size=dataset_config.size,
            dim=dataset_config.dim,
            metric_type=metric_type_map(dataset_config.metric_type),
            use_shuffled=dataset_config.use_shuffled,
            with_gt=dataset_config.with_gt,
            dir=dataset_config.dir,
            file_num=dataset_config.file_count,
        )
        super().__init__(
            name=name,
            description=description,
            load_timeout=load_timeout,
            optimize_timeout=optimize_timeout,
            dataset=DatasetManager(data=dataset),
        )


type2case = {
    CaseType.CapacityDim960: CapacityDim960,
    CaseType.CapacityDim128: CapacityDim128,
    CaseType.Performance768D100M: Performance768D100M,
    CaseType.Performance768D10M: Performance768D10M,
    CaseType.Performance768D1M: Performance768D1M,
    CaseType.Performance768D10M1P: Performance768D10M1P,
    CaseType.Performance768D1M1P: Performance768D1M1P,
    CaseType.Performance768D10M99P: Performance768D10M99P,
    CaseType.Performance768D1M99P: Performance768D1M99P,
    CaseType.Performance1536D500K: Performance1536D500K,
    CaseType.Performance1536D5M: Performance1536D5M,
    CaseType.Performance1536D500K1P: Performance1536D500K1P,
    CaseType.Performance1536D5M1P: Performance1536D5M1P,
    CaseType.Performance1536D500K99P: Performance1536D500K99P,
    CaseType.Performance1536D5M99P: Performance1536D5M99P,
    CaseType.Performance1536D50K: Performance1536D50K,
    CaseType.PerformanceCustomDataset: PerformanceCustomDataset,
    CaseType.LabelFilterPerformanceCase: LabelFilterPerformanceCase,
}
