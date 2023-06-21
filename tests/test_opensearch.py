import logging
from vector_db_bench.backend.clients import DB
from vector_db_bench.backend.clients.api import MetricType
from vector_db_bench.backend.clients import OpenSearch
from vector_db_bench.backend.clients.opensearch.config import (
    OpensearchConfig,
    OpensearchIndexConfig,
)

import numpy as np


log = logging.getLogger(__name__)

project_name = ""
mc_endpoint = ""
access_id = ""
access_key = ""

os_endpoint = ""
instance_id = ""
access_user_name = ""
access_password = ""

dim = 768


class TestModels:
    def test_insert_and_search(self):
        assert DB.OpenSearch.init_cls == OpenSearch
        assert DB.OpenSearch.init_cls.config_cls() == OpensearchConfig
        assert DB.OpenSearch.init_cls.case_config_cls() == OpensearchIndexConfig

        dbcls = DB.OpenSearch.init_cls
        dbConfig = dbcls.config_cls()(
            project_name=project_name,
            mc_endpoint=mc_endpoint,
            access_id=access_id,
            access_key=access_key,
            os_endpoint=os_endpoint,
            instance_id=instance_id,
            access_user_name=access_user_name,
            access_password=access_password,
        )
        dbCaseConfig = dbcls.case_config_cls()(metric_type=MetricType.L2, ef=500)

        os = dbcls(
            dim=dim,
            db_config=dbConfig.to_dict(),
            db_case_config=dbCaseConfig,
            drop_old=False,
        )

        count = 10_000
        filter_rate = 0.9
        embeddings = [[0.5 + np.random.random() * 0.1 for _ in range(dim)] for _ in range(count)]

        # # insert
        # with es.init():
        #     res = es.insert_embeddings(embeddings=embeddings, metadata=range(count))
        #     # bulk_insert return
        #     assert (
        #         res == count
        #     ), f"the return count of bulk insert ({res}) is not equal to count ({count})"

        #     # indice_count return
        #     es.client.indices.refresh()
        #     esCountRes = es.client.count(index=es.indice)
        #     countResCount = esCountRes.raw["count"]
        #     assert (
        #         countResCount == count
        #     ), f"the return count of es client ({countResCount}) is not equal to count ({count})"

        # search res format
        with os.init():
            test_id = np.random.randint(count)
            log.info(f"test_id: {test_id}")
            q = embeddings[test_id]
            k = 100
            res = os.search_embedding(query=q, k=k)
            log.info(f"search_results_id: {res}")
            assert len(res) == k, f"the length of res is not equal to k"
            assert type(res[0]) == int, f"the res type is not int[]"
            
            assert None

        # search
        # with os.init():
        #     test_id = np.random.randint(count)
        #     log.info(f"test_id: {test_id}")
        #     q = embeddings[test_id]

        #     res = os.search_embedding(query=q, k=100)
        #     log.info(f"search_results_id: {res}")
        #     assert (
        #         res[0] == test_id
        #     ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"

        # # search with filters
        # with es.init():
        #     test_id = np.random.randint(count * filter_rate, count)
        #     log.info(f"test_id: {test_id}")
        #     q = embeddings[test_id]

        #     res = es.search_embedding(
        #         query=q, k=100, filters={"id": count * filter_rate}
        #     )
        #     log.info(f"search_results_id: {res}")
        #     assert (
        #         res[0] == test_id
        #     ), f"the most nearest neighbor ({res[0]}) id is not test_id ({test_id})"
        #     isFilter = True
        #     for id in res:
        #         if id < count * filter_rate:
        #             isFilter = False
        #             break
        #     assert isFilter, f"filters failed"
