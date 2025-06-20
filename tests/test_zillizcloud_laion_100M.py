from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig, ZillizCloudConfig
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.models import CaseConfig, ConcurrencySearchConfig, TaskConfig


# 填一下zillizcloud的账号信息
uri = ""  # 尽量使用private_link，减少公网的干扰
user = ""
password = ""


# 并发测试的规模，可以自行调整
num_concurrency = [100]  # 并发数
concurrency_duration = 60 * 5  # 并发测试时长(s)


# 不要改这部分参数配置
case_id = CaseType.Performance768D100M  # Laion_100M 数据集
use_aliyun = True  # if false，会从美西下载数据集，速度较慢
drop_old = False  # if true，测试时会把我们已经import好了的数据删掉
task_label = "dewu_test"


if __name__ == "__main__":
    runner = BenchMarkRunner()

    db_config = ZillizCloudConfig(
        uri=uri,
        user=user,
        password=password,
    )
    db_case_config = AutoIndexConfig(level=1)
    case_config = CaseConfig(
        case_id=case_id,
    )
    case_config.concurrency_search_config = ConcurrencySearchConfig(
        concurrency_duration=concurrency_duration,
        num_concurrency=num_concurrency,
    )

    task = TaskConfig(
        db=DB.ZillizCloud,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=case_config,
    )

    runner.set_download_address(use_aliyun)
    runner.set_drop_old(drop_old)
    runner.run(tasks=[task], task_label=task_label)
