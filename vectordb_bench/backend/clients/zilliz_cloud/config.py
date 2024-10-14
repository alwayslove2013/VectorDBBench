from pydantic import SecretStr

from ..api import DBCaseConfig, DBConfig
from ..milvus.config import MilvusIndexConfig, IndexType


class ZillizCloudConfig(DBConfig):
    uri: SecretStr = (
        "https://in01-9024610b4458ba9.aws-us-west-2.vectordb-uat3.zillizcloud.com:19530"
    )
    user: str = "db_admin"
    password: SecretStr = "Milvus123"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user,
            "password": self.password.get_secret_value(),
        }


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX
    level: int = 1

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "level": self.level,
            },
        }
