from opensearchpy import OpenSearch
import numpy as np
import time

host = "vpc-tianmin-test-aotn2gd2o7hrrtqeoikxyo2rgq.us-west-2.es.amazonaws.com"
port = 443
auth = (
    "admin",
    "ahh4Eevee2aa7diuTahh8oos@",
)  # For testing only. Don't store credentials in code.
# ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.

# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    http_compress=True,  # enables gzip compression for request bodies
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    # ca_certs = ca_certs_path
)
index_name: str = "vdb_bench_index"

query = np.random.rand(768).tolist()

k = 5
vector_col_name: str = "embedding"
id_col_name: str = "id"
filters = {"id": int(1_000_000 * 0.99)}
filters = None
body = {
    "size": k,
    "query": {"knn": {vector_col_name: {"vector": query, "k": k}}},
    **({"filter": {"range": {id_col_name: {"gt": filters["id"]}}}} if filters else {}),
}
resp = client.search(
    index=index_name,
    body=body,
    size=k,
    _source=False,
    docvalue_fields=[id_col_name],
    stored_fields="_none_",
    filter_path=[f"hits.hits.fields.{id_col_name}"],
)
resp

s = time.perf_counter()
try:
    resp = client.search(
        index=index_name,
        body=body,
        size=k,
        _source=False,
        docvalue_fields=[id_col_name],
        stored_fields="_none_",
        filter_path=[f"hits.hits.fields.{id_col_name}"],
        timeout=600,
    )
    print(resp)
except Exception as e:
    print(e)


print(time.perf_counter() - s)

client.indices.refresh(index=index_name)

ef = 150
settings = {
    "index": {
        "knn": {
            "algo_param": {
                "ef_search": ef,
            },
        }
    }
}
client.indices.put_settings(index=index_name, body=settings)
client.indices.get_settings(index=index_name)
client.indices.refresh(index=index_name)

k = 100
body = {
    "size": k,
    "query": {
        "knn": {vector_col_name: {"vector": query, "k": k}},
    },
}
client.search(
    index=index_name,
    body=body,
    size=k,
    _source=False,
    docvalue_fields=[id_col_name],
    stored_fields="_none_",
    filter_path=[f"hits.hits.fields.{id_col_name}"],
)

client.tasks.get(task_id=force_merge_task_id)
{
    "completed": False,
    "task": {
        "node": "xrhDYtPmQpWHxxGnjJjp3w",
        "id": 25372391,
        "type": "transport",
        "action": "indices:admin/forcemerge",
        "description": "Force-merge indices [vdb_bench_index], maxSegments[1], onlyExpungeDeletes[false], flush[true], primaryOnly[false]",
        "start_time_in_millis": 1733473592305,
        "running_time_in_nanos": 272943945187,
        "cancellable": False,
        "cancelled": False,
        "headers": {},
        "resource_stats": {
            "average": {"cpu_time_in_nanos": 0, "memory_in_bytes": 0},
            "total": {"cpu_time_in_nanos": 0, "memory_in_bytes": 0},
            "min": {"cpu_time_in_nanos": 0, "memory_in_bytes": 0},
            "max": {"cpu_time_in_nanos": 0, "memory_in_bytes": 0},
            "thread_info": {"thread_executions": 0, "active_threads": 0},
        },
    },
}

{
    "_shards": {"total": 10, "successful": 5, "failed": 0},
    "_all": {
        "primaries": {
            "docs": {"count": 1000000, "deleted": 0},
            "store": {"size_in_bytes": 18338974934, "reserved_in_bytes": 0},
            "indexing": {
                "index_total": 1000000,
                "index_time_in_millis": 419080,
                "index_current": 0,
                "index_failed": 0,
                "delete_total": 0,
                "delete_time_in_millis": 0,
                "delete_current": 0,
                "noop_update_total": 0,
                "is_throttled": False,
                "throttle_time_in_millis": 0,
                "doc_status": {},
            },
            "get": {
                "total": 0,
                "time_in_millis": 0,
                "exists_total": 0,
                "exists_time_in_millis": 0,
                "missing_total": 0,
                "missing_time_in_millis": 0,
                "current": 0,
            },
            "search": {
                "open_contexts": 0,
                "query_total": 4358220,
                "query_time_in_millis": 10240103,
                "query_current": 0,
                "concurrent_query_total": 4358220,
                "concurrent_query_time_in_millis": 10240103,
                "concurrent_query_current": 0,
                "concurrent_avg_slice_count": 1.0,
                "fetch_total": 4358220,
                "fetch_time_in_millis": 167217,
                "fetch_current": 0,
                "scroll_total": 0,
                "scroll_time_in_millis": 0,
                "scroll_current": 0,
                "point_in_time_total": 0,
                "point_in_time_time_in_millis": 0,
                "point_in_time_current": 0,
                "suggest_total": 0,
                "suggest_time_in_millis": 0,
                "suggest_current": 0,
                "search_idle_reactivate_count_total": 0,
            },
            "merges": {
                "current": 0,
                "current_docs": 0,
                "current_size_in_bytes": 0,
                "total": 15,
                "total_time_in_millis": 2771749,
                "total_docs": 1007699,
                "total_size_in_bytes": 18479831226,
                "total_stopped_time_in_millis": 0,
                "total_throttled_time_in_millis": 0,
                "total_auto_throttle_in_bytes": 95325090,
                "unreferenced_file_cleanups_performed": 0,
            },
            "refresh": {
                "total": 199,
                "total_time_in_millis": 313473,
                "external_total": 164,
                "external_total_time_in_millis": 217572,
                "listeners": 0,
            },
            "flush": {"total": 30, "periodic": 30, "total_time_in_millis": 1853345},
            "warmer": {"current": 0, "total": 154, "total_time_in_millis": 0},
            "query_cache": {
                "memory_size_in_bytes": 0,
                "total_count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "cache_size": 0,
                "cache_count": 0,
                "evictions": 0,
            },
            "fielddata": {"memory_size_in_bytes": 0, "evictions": 0},
            "completion": {"size_in_bytes": 0},
            "segments": {
                "count": 5,
                "memory_in_bytes": 0,
                "terms_memory_in_bytes": 0,
                "stored_fields_memory_in_bytes": 0,
                "term_vectors_memory_in_bytes": 0,
                "norms_memory_in_bytes": 0,
                "points_memory_in_bytes": 0,
                "doc_values_memory_in_bytes": 0,
                "index_writer_memory_in_bytes": 0,
                "version_map_memory_in_bytes": 0,
                "fixed_bit_set_memory_in_bytes": 0,
                "max_unsafe_auto_id_timestamp": -1,
                "remote_store": {
                    "upload": {
                        "total_upload_size": {
                            "started_bytes": 0,
                            "succeeded_bytes": 0,
                            "failed_bytes": 0,
                        },
                        "refresh_size_lag": {"total_bytes": 0, "max_bytes": 0},
                        "max_refresh_time_lag_in_millis": 0,
                        "total_time_spent_in_millis": 0,
                        "pressure": {"total_rejections": 0},
                    },
                    "download": {
                        "total_download_size": {
                            "started_bytes": 0,
                            "succeeded_bytes": 0,
                            "failed_bytes": 0,
                        },
                        "total_time_spent_in_millis": 0,
                    },
                },
                "segment_replication": {
                    "max_bytes_behind": 0,
                    "total_bytes_behind": 0,
                    "max_replication_lag": 0,
                },
                "file_sizes": {},
            },
            "translog": {
                "operations": 0,
                "size_in_bytes": 275,
                "uncommitted_operations": 0,
                "uncommitted_size_in_bytes": 275,
                "earliest_last_modified_age": 5117621,
                "remote_store": {
                    "upload": {
                        "total_uploads": {"started": 0, "failed": 0, "succeeded": 0},
                        "total_upload_size": {
                            "started_bytes": 0,
                            "failed_bytes": 0,
                            "succeeded_bytes": 0,
                        },
                    }
                },
            },
            "request_cache": {
                "memory_size_in_bytes": 0,
                "evictions": 0,
                "hit_count": 0,
                "miss_count": 0,
            },
            "recovery": {
                "current_as_source": 0,
                "current_as_target": 0,
                "throttle_time_in_millis": 0,
            },
        },
        "total": {
            "docs": {"count": 1000000, "deleted": 0},
            "store": {"size_in_bytes": 18338974934, "reserved_in_bytes": 0},
            "indexing": {
                "index_total": 1000000,
                "index_time_in_millis": 419080,
                "index_current": 0,
                "index_failed": 0,
                "delete_total": 0,
                "delete_time_in_millis": 0,
                "delete_current": 0,
                "noop_update_total": 0,
                "is_throttled": False,
                "throttle_time_in_millis": 0,
                "doc_status": {},
            },
            "get": {
                "total": 0,
                "time_in_millis": 0,
                "exists_total": 0,
                "exists_time_in_millis": 0,
                "missing_total": 0,
                "missing_time_in_millis": 0,
                "current": 0,
            },
            "search": {
                "open_contexts": 0,
                "query_total": 4358220,
                "query_time_in_millis": 10240103,
                "query_current": 0,
                "concurrent_query_total": 4358220,
                "concurrent_query_time_in_millis": 10240103,
                "concurrent_query_current": 0,
                "concurrent_avg_slice_count": 1.0,
                "fetch_total": 4358220,
                "fetch_time_in_millis": 167217,
                "fetch_current": 0,
                "scroll_total": 0,
                "scroll_time_in_millis": 0,
                "scroll_current": 0,
                "point_in_time_total": 0,
                "point_in_time_time_in_millis": 0,
                "point_in_time_current": 0,
                "suggest_total": 0,
                "suggest_time_in_millis": 0,
                "suggest_current": 0,
                "search_idle_reactivate_count_total": 0,
            },
            "merges": {
                "current": 0,
                "current_docs": 0,
                "current_size_in_bytes": 0,
                "total": 15,
                "total_time_in_millis": 2771749,
                "total_docs": 1007699,
                "total_size_in_bytes": 18479831226,
                "total_stopped_time_in_millis": 0,
                "total_throttled_time_in_millis": 0,
                "total_auto_throttle_in_bytes": 95325090,
                "unreferenced_file_cleanups_performed": 0,
            },
            "refresh": {
                "total": 199,
                "total_time_in_millis": 313473,
                "external_total": 164,
                "external_total_time_in_millis": 217572,
                "listeners": 0,
            },
            "flush": {"total": 30, "periodic": 30, "total_time_in_millis": 1853345},
            "warmer": {"current": 0, "total": 154, "total_time_in_millis": 0},
            "query_cache": {
                "memory_size_in_bytes": 0,
                "total_count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "cache_size": 0,
                "cache_count": 0,
                "evictions": 0,
            },
            "fielddata": {"memory_size_in_bytes": 0, "evictions": 0},
            "completion": {"size_in_bytes": 0},
            "segments": {
                "count": 5,
                "memory_in_bytes": 0,
                "terms_memory_in_bytes": 0,
                "stored_fields_memory_in_bytes": 0,
                "term_vectors_memory_in_bytes": 0,
                "norms_memory_in_bytes": 0,
                "points_memory_in_bytes": 0,
                "doc_values_memory_in_bytes": 0,
                "index_writer_memory_in_bytes": 0,
                "version_map_memory_in_bytes": 0,
                "fixed_bit_set_memory_in_bytes": 0,
                "max_unsafe_auto_id_timestamp": -1,
                "remote_store": {
                    "upload": {
                        "total_upload_size": {
                            "started_bytes": 0,
                            "succeeded_bytes": 0,
                            "failed_bytes": 0,
                        },
                        "refresh_size_lag": {"total_bytes": 0, "max_bytes": 0},
                        "max_refresh_time_lag_in_millis": 0,
                        "total_time_spent_in_millis": 0,
                        "pressure": {"total_rejections": 0},
                    },
                    "download": {
                        "total_download_size": {
                            "started_bytes": 0,
                            "succeeded_bytes": 0,
                            "failed_bytes": 0,
                        },
                        "total_time_spent_in_millis": 0,
                    },
                },
                "segment_replication": {
                    "max_bytes_behind": 0,
                    "total_bytes_behind": 0,
                    "max_replication_lag": 0,
                },
                "file_sizes": {},
            },
            "translog": {
                "operations": 0,
                "size_in_bytes": 275,
                "uncommitted_operations": 0,
                "uncommitted_size_in_bytes": 275,
                "earliest_last_modified_age": 5117621,
                "remote_store": {
                    "upload": {
                        "total_uploads": {"started": 0, "failed": 0, "succeeded": 0},
                        "total_upload_size": {
                            "started_bytes": 0,
                            "failed_bytes": 0,
                            "succeeded_bytes": 0,
                        },
                    }
                },
            },
            "request_cache": {
                "memory_size_in_bytes": 0,
                "evictions": 0,
                "hit_count": 0,
                "miss_count": 0,
            },
            "recovery": {
                "current_as_source": 0,
                "current_as_target": 0,
                "throttle_time_in_millis": 0,
            },
        },
    },
    "indices": {
        "vdb_bench_index": {
            "uuid": "FHSVd9w5Qo6Cpwy5lMLNgQ",
            "primaries": {
                "docs": {"count": 1000000, "deleted": 0},
                "store": {"size_in_bytes": 18338974934, "reserved_in_bytes": 0},
                "indexing": {
                    "index_total": 1000000,
                    "index_time_in_millis": 419080,
                    "index_current": 0,
                    "index_failed": 0,
                    "delete_total": 0,
                    "delete_time_in_millis": 0,
                    "delete_current": 0,
                    "noop_update_total": 0,
                    "is_throttled": False,
                    "throttle_time_in_millis": 0,
                    "doc_status": {},
                },
                "get": {
                    "total": 0,
                    "time_in_millis": 0,
                    "exists_total": 0,
                    "exists_time_in_millis": 0,
                    "missing_total": 0,
                    "missing_time_in_millis": 0,
                    "current": 0,
                },
                "search": {
                    "open_contexts": 0,
                    "query_total": 4358220,
                    "query_time_in_millis": 10240103,
                    "query_current": 0,
                    "concurrent_query_total": 4358220,
                    "concurrent_query_time_in_millis": 10240103,
                    "concurrent_query_current": 0,
                    "concurrent_avg_slice_count": 1.0,
                    "fetch_total": 4358220,
                    "fetch_time_in_millis": 167217,
                    "fetch_current": 0,
                    "scroll_total": 0,
                    "scroll_time_in_millis": 0,
                    "scroll_current": 0,
                    "point_in_time_total": 0,
                    "point_in_time_time_in_millis": 0,
                    "point_in_time_current": 0,
                    "suggest_total": 0,
                    "suggest_time_in_millis": 0,
                    "suggest_current": 0,
                    "search_idle_reactivate_count_total": 0,
                },
                "merges": {
                    "current": 0,
                    "current_docs": 0,
                    "current_size_in_bytes": 0,
                    "total": 15,
                    "total_time_in_millis": 2771749,
                    "total_docs": 1007699,
                    "total_size_in_bytes": 18479831226,
                    "total_stopped_time_in_millis": 0,
                    "total_throttled_time_in_millis": 0,
                    "total_auto_throttle_in_bytes": 95325090,
                    "unreferenced_file_cleanups_performed": 0,
                },
                "refresh": {
                    "total": 199,
                    "total_time_in_millis": 313473,
                    "external_total": 164,
                    "external_total_time_in_millis": 217572,
                    "listeners": 0,
                },
                "flush": {"total": 30, "periodic": 30, "total_time_in_millis": 1853345},
                "warmer": {"current": 0, "total": 154, "total_time_in_millis": 0},
                "query_cache": {
                    "memory_size_in_bytes": 0,
                    "total_count": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "cache_size": 0,
                    "cache_count": 0,
                    "evictions": 0,
                },
                "fielddata": {"memory_size_in_bytes": 0, "evictions": 0},
                "completion": {"size_in_bytes": 0},
                "segments": {
                    "count": 5,
                    "memory_in_bytes": 0,
                    "terms_memory_in_bytes": 0,
                    "stored_fields_memory_in_bytes": 0,
                    "term_vectors_memory_in_bytes": 0,
                    "norms_memory_in_bytes": 0,
                    "points_memory_in_bytes": 0,
                    "doc_values_memory_in_bytes": 0,
                    "index_writer_memory_in_bytes": 0,
                    "version_map_memory_in_bytes": 0,
                    "fixed_bit_set_memory_in_bytes": 0,
                    "max_unsafe_auto_id_timestamp": -1,
                    "remote_store": {
                        "upload": {
                            "total_upload_size": {
                                "started_bytes": 0,
                                "succeeded_bytes": 0,
                                "failed_bytes": 0,
                            },
                            "refresh_size_lag": {"total_bytes": 0, "max_bytes": 0},
                            "max_refresh_time_lag_in_millis": 0,
                            "total_time_spent_in_millis": 0,
                            "pressure": {"total_rejections": 0},
                        },
                        "download": {
                            "total_download_size": {
                                "started_bytes": 0,
                                "succeeded_bytes": 0,
                                "failed_bytes": 0,
                            },
                            "total_time_spent_in_millis": 0,
                        },
                    },
                    "segment_replication": {
                        "max_bytes_behind": 0,
                        "total_bytes_behind": 0,
                        "max_replication_lag": 0,
                    },
                    "file_sizes": {},
                },
                "translog": {
                    "operations": 0,
                    "size_in_bytes": 275,
                    "uncommitted_operations": 0,
                    "uncommitted_size_in_bytes": 275,
                    "earliest_last_modified_age": 5117621,
                    "remote_store": {
                        "upload": {
                            "total_uploads": {
                                "started": 0,
                                "failed": 0,
                                "succeeded": 0,
                            },
                            "total_upload_size": {
                                "started_bytes": 0,
                                "failed_bytes": 0,
                                "succeeded_bytes": 0,
                            },
                        }
                    },
                },
                "request_cache": {
                    "memory_size_in_bytes": 0,
                    "evictions": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                },
                "recovery": {
                    "current_as_source": 0,
                    "current_as_target": 0,
                    "throttle_time_in_millis": 0,
                },
            },
            "total": {
                "docs": {"count": 1000000, "deleted": 0},
                "store": {"size_in_bytes": 18338974934, "reserved_in_bytes": 0},
                "indexing": {
                    "index_total": 1000000,
                    "index_time_in_millis": 419080,
                    "index_current": 0,
                    "index_failed": 0,
                    "delete_total": 0,
                    "delete_time_in_millis": 0,
                    "delete_current": 0,
                    "noop_update_total": 0,
                    "is_throttled": False,
                    "throttle_time_in_millis": 0,
                    "doc_status": {},
                },
                "get": {
                    "total": 0,
                    "time_in_millis": 0,
                    "exists_total": 0,
                    "exists_time_in_millis": 0,
                    "missing_total": 0,
                    "missing_time_in_millis": 0,
                    "current": 0,
                },
                "search": {
                    "open_contexts": 0,
                    "query_total": 4358220,
                    "query_time_in_millis": 10240103,
                    "query_current": 0,
                    "concurrent_query_total": 4358220,
                    "concurrent_query_time_in_millis": 10240103,
                    "concurrent_query_current": 0,
                    "concurrent_avg_slice_count": 1.0,
                    "fetch_total": 4358220,
                    "fetch_time_in_millis": 167217,
                    "fetch_current": 0,
                    "scroll_total": 0,
                    "scroll_time_in_millis": 0,
                    "scroll_current": 0,
                    "point_in_time_total": 0,
                    "point_in_time_time_in_millis": 0,
                    "point_in_time_current": 0,
                    "suggest_total": 0,
                    "suggest_time_in_millis": 0,
                    "suggest_current": 0,
                    "search_idle_reactivate_count_total": 0,
                },
                "merges": {
                    "current": 0,
                    "current_docs": 0,
                    "current_size_in_bytes": 0,
                    "total": 15,
                    "total_time_in_millis": 2771749,
                    "total_docs": 1007699,
                    "total_size_in_bytes": 18479831226,
                    "total_stopped_time_in_millis": 0,
                    "total_throttled_time_in_millis": 0,
                    "total_auto_throttle_in_bytes": 95325090,
                    "unreferenced_file_cleanups_performed": 0,
                },
                "refresh": {
                    "total": 199,
                    "total_time_in_millis": 313473,
                    "external_total": 164,
                    "external_total_time_in_millis": 217572,
                    "listeners": 0,
                },
                "flush": {"total": 30, "periodic": 30, "total_time_in_millis": 1853345},
                "warmer": {"current": 0, "total": 154, "total_time_in_millis": 0},
                "query_cache": {
                    "memory_size_in_bytes": 0,
                    "total_count": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                    "cache_size": 0,
                    "cache_count": 0,
                    "evictions": 0,
                },
                "fielddata": {"memory_size_in_bytes": 0, "evictions": 0},
                "completion": {"size_in_bytes": 0},
                "segments": {
                    "count": 5,
                    "memory_in_bytes": 0,
                    "terms_memory_in_bytes": 0,
                    "stored_fields_memory_in_bytes": 0,
                    "term_vectors_memory_in_bytes": 0,
                    "norms_memory_in_bytes": 0,
                    "points_memory_in_bytes": 0,
                    "doc_values_memory_in_bytes": 0,
                    "index_writer_memory_in_bytes": 0,
                    "version_map_memory_in_bytes": 0,
                    "fixed_bit_set_memory_in_bytes": 0,
                    "max_unsafe_auto_id_timestamp": -1,
                    "remote_store": {
                        "upload": {
                            "total_upload_size": {
                                "started_bytes": 0,
                                "succeeded_bytes": 0,
                                "failed_bytes": 0,
                            },
                            "refresh_size_lag": {"total_bytes": 0, "max_bytes": 0},
                            "max_refresh_time_lag_in_millis": 0,
                            "total_time_spent_in_millis": 0,
                            "pressure": {"total_rejections": 0},
                        },
                        "download": {
                            "total_download_size": {
                                "started_bytes": 0,
                                "succeeded_bytes": 0,
                                "failed_bytes": 0,
                            },
                            "total_time_spent_in_millis": 0,
                        },
                    },
                    "segment_replication": {
                        "max_bytes_behind": 0,
                        "total_bytes_behind": 0,
                        "max_replication_lag": 0,
                    },
                    "file_sizes": {},
                },
                "translog": {
                    "operations": 0,
                    "size_in_bytes": 275,
                    "uncommitted_operations": 0,
                    "uncommitted_size_in_bytes": 275,
                    "earliest_last_modified_age": 5117621,
                    "remote_store": {
                        "upload": {
                            "total_uploads": {
                                "started": 0,
                                "failed": 0,
                                "succeeded": 0,
                            },
                            "total_upload_size": {
                                "started_bytes": 0,
                                "failed_bytes": 0,
                                "succeeded_bytes": 0,
                            },
                        }
                    },
                },
                "request_cache": {
                    "memory_size_in_bytes": 0,
                    "evictions": 0,
                    "hit_count": 0,
                    "miss_count": 0,
                },
                "recovery": {
                    "current_as_source": 0,
                    "current_as_target": 0,
                    "throttle_time_in_millis": 0,
                },
            },
        }
    },
}
