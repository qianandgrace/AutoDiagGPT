import re
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter


def detect_code(query):
    match = re.search(r'P\d{4}', query)
    return match.group(0) if match else None


def retrieve(index, query):
    code = detect_code(query)
     
    # 👉 优先故障码
    if code:
        retriever = index.as_retriever(
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="code", value=code)]
            ),
            similarity_top_k=3
        )
        return retriever.retrieve(query)

    # 👉 分路召回
    retriever_obd = index.as_retriever(
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="type", value="obd")]
        ),
        similarity_top_k=3
    )

    retriever_case = index.as_retriever(
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="type", value="case")]
        ),
        similarity_top_k=2
    )

    nodes = retriever_obd.retrieve(query) + retriever_case.retrieve(query)
    return nodes


if __name__ == "__main__":

    query = "我的车报了P1073"
    print(detect_code(query))