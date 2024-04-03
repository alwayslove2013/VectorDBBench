from pymilvus import Collection, utility, connections

uri = "http://10.15.11.207:19530"
connections.connect(uri=uri)

collection_name = "VectorDBBenchCollection"
coll = Collection(collection_name)

utility.get_query_segment_info(collection_name)
segments = utility.get_query_segment_info(collection_name)
[seg.num_rows for seg in segments]