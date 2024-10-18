from ..base import BaseModel
from enum import StrEnum


class FilterType(StrEnum):
    Int = "Int"  # test ">="
    Label = "Label"  # test "=="
    NonFilter = "NonFilter"


class Filter(BaseModel):
    type: FilterType
    filter_rate: float = 0.0

    def get_groundtruth_file(self) -> str:
        raise NotImplementedError


class NonFilter(Filter):
    type: FilterType = FilterType.NonFilter
    filter_rate: float = 0.0

    def get_groundtruth_file(self) -> str:
        return "neighbors.parquet"


class IntFilter(Filter):
    """
    filter expr: id >= count * filter_rate
    """

    type: FilterType = FilterType.Int
    int_field: str = "id"
    int_value: int

    def get_groundtruth_file(self) -> str:
        if self.filter_rate == 0.01:
            return "neighbors_head_1p.parquet"
        if self.filter_rate == 0.99:
            return "neighbors_tail_1p.parquet"
        raise RuntimeError(f"Not Support Int Filter - {self.filter_rate}")


class LabelFilter(Filter):
    """
    filter expr: label_field == label_value, like `color == "red"`
    """

    type: FilterType = FilterType.Label
    label_field: str = "labels"
    label_percentage: float

    @property
    def label_value(self) -> str:
        p = self.label_percentage * 100
        if p >= 1:
            return f"label_{int(p)}p"  # such as 5p, 20p, 1p, ...
        else:
            return f"label_{p:.1f}p"  # such as 0.1p, 0.5p, ...

    def __init__(self, label_percentage: float, **kwargs):
        filter_rate = 1.0 - label_percentage
        super().__init__(
            filter_rate=filter_rate, label_percentage=label_percentage, **kwargs
        )

    def get_groundtruth_file(self) -> str:
        return f"neighbors_{self.label_field}_{self.label_value}.parquet"


MSMarcoV2_Percentage_To_Label = {
    0.01: "f8d5a851-0fc1-4f84-808c-450721a-20",
    0.05: "f8d5a851-0fc1-4f84-808c-450721a-21",
    0.1: "f8d5a851-0fc1-4f84-808c-450721a-22",
    0.2: "f8d5a851-0fc1-4f84-808c-450721a-23",
    0.4: "f8d5a851-0fc1-4f84-808c-450721a-24",
}


class MSMarcoV2LabelFilter(LabelFilter):
    """
    only for msmarco_v2 dataset.
    filter expr: label_field == label_value, like `color == "red"`
    """

    type: FilterType = FilterType.Label
    label_field: str = "label"
    label_percentage: float

    @property
    def label_value(self) -> str:
        if self.label_percentage not in MSMarcoV2_Percentage_To_Label:
            raise KeyError(
                f"Can not find groundtruth file for percentage-{self.label_percentage} in MSMarcoV2 dataset"
            )
        return MSMarcoV2_Percentage_To_Label.get(self.label_percentage)

    def __init__(self, label_percentage: float, **kwargs):
        super().__init__(label_percentage=label_percentage, **kwargs)

    def get_groundtruth_file(self) -> str:
        return f"neighbors_{self.label_field}_{self.label_value}.parquet"
