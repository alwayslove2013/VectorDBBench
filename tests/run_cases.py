from vectordb_bench import config
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.config import MilvusConfig
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig, ZillizCloudConfig
from vectordb_bench.backend.dataset import DatasetWithSizeType
from vectordb_bench.frontend.config.dbCaseConfigs import generate_label_filter_cases
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import CaseConfig, CaseConfigParamType, ConcurrencySearchConfig, TaskConfig
import time


uri = ""
user = ""
password = ""

db_label = f"8cu-perf"
version = ""


def generate_tasks(
    tasks: list[TaskConfig], db_label: str, case_configs: list[CaseConfig], db_case_config: AutoIndexConfig
):
    db_config = ZillizCloudConfig(uri=uri, user=user, password=password, db_label=db_label, version=version)
    for case_config in case_configs:
        tasks.append(
            TaskConfig(db=DB.ZillizCloud, db_config=db_config, case_config=case_config, db_case_config=db_case_config)
        )


def get_static_cases(data_size: DatasetWithSizeType):
    if data_size == DatasetWithSizeType.CohereMedium:
        return [CaseConfig(case_id=CaseType.Performance768D1M)]
    elif data_size == DatasetWithSizeType.CohereLarge:
        return [CaseConfig(case_id=CaseType.Performance768D10M)]
    elif data_size == DatasetWithSizeType.OpenAILarge:
        return [CaseConfig(case_id=CaseType.Performance1536D5M)]
    elif data_size == DatasetWithSizeType.OpenAIMedium:
        return [CaseConfig(case_id=CaseType.Performance1536D500K)]
    elif data_size == DatasetWithSizeType.BioasqLarge:
        return [CaseConfig(case_id=CaseType.Performance1024D10M)]
    raise RuntimeError(f"not support data: {data_size}")


def test_static_level(
    runner: BenchMarkRunner,
    data_size: DatasetWithSizeType,
    levels: list[int],
    num_concurrency: list[int] = config.NUM_CONCURRENCY,
    concurrency_duration: int = config.CONCURRENCY_DURATION,
    k: int = 100,
    use_partition_key: bool = False,
    num_partitions: int = 1,
):
    db_label_ = f"{db_label}{f'-par{num_partitions}' if use_partition_key else ''}"
    while runner.has_running():
        time.sleep(60)
    tasks: list[TaskConfig] = []

    case_configs = get_static_cases(data_size)
    for case_config in case_configs:
        case_config.concurrency_search_config.num_concurrency = num_concurrency
        case_config.concurrency_search_config.concurrency_duration = concurrency_duration
        case_config.k = k
    for level in levels:
        db_case_config = AutoIndexConfig(
            level=level, use_partition_key=use_partition_key, num_partitions=num_partitions
        )
        db_label__ = f"{db_label_}-level-{level}"
        generate_tasks(tasks, db_label__, case_configs, db_case_config)

    task_label = f"{db_label_}-static-{data_size.name}-level-{levels}"
    runner.set_drop_old(False)
    runner.run(tasks=tasks, task_label=task_label)


if __name__ == "__main__":
    runner = BenchMarkRunner()
    data_size = DatasetWithSizeType.CohereLarge
    levels = [1]
    num_concurrency = [200]
    concurrency_duration = 60 * 10
    k = 100
    use_partition_key = False
    num_partitions = 1
    test_static_level(
        runner=runner,
        data_size=data_size,
        levels=levels,
        num_concurrency=num_concurrency,
        concurrency_duration=concurrency_duration,
    )
