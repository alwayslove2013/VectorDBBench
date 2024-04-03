from vectordb_bench.backend.clients.milvus.config import GPUCAGRAConfig, MilvusConfig
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList, dbConfigs, activedCaseList, allCaseConfigs):
    # tasks = []
    # for db in activedDbList:
    #     for case in activedCaseList:
    #         task = TaskConfig(
    #                 db=db.value,
    #                 db_config=dbConfigs[db],
    #                 case_config=CaseConfig(
    #                     case_id=case.value,
    #                     custom_case={},
    #                 ),
    #                 db_case_config=db.case_config_cls(
    #                     allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None)
    #                 )(**{key.value: value for key, value in allCaseConfigs[db][case].items()}),
    #             )
    #         tasks.append(task)
            
    tasks = []
    max_iterations_list = [16, 18, 20, 22]
    itopk_size_list = [100, 105, 110, 115, 120, 125, 130]
    search_width_list = [1]
    nn = 100
    for db in activedDbList:
        for case in activedCaseList:
            for max_iterations in max_iterations_list:
                for itopk_size in itopk_size_list:
                    for search_width in search_width_list:
                        db_config = MilvusConfig(uri="http://10.15.11.207:19530", db_label=f"openai-iter-{max_iterations}-ef-{itopk_size}-sw-{search_width}-nn-{nn}")
                        db_case_config = GPUCAGRAConfig(search_width=search_width, max_iterations=max_iterations,itopk_size=itopk_size)
                        task = TaskConfig(
                                db=db.value,
                                db_config=db_config,
                                case_config=CaseConfig(
                                    case_id=case.value,
                                    custom_case={},
                                ),
                                db_case_config=db_case_config,
                            )
                        tasks.append(task)
    
    return tasks
