from pydantic import SecretStr
from ..api import DBConfig


class PineconeConfig(DBConfig):
    api_key: SecretStr = "cd6ba5ba-bb3f-4ff7-863a-56a83e5667a8"
    environment: SecretStr = "us-west1-gcp"
    index_name: str = "minmin"

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "environment": self.environment.get_secret_value(),
            "index_name": self.index_name,
        }
