from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)

def rerank(query, nodes, top_k=4):
    pairs = [[query, n.text] for n in nodes]
    scores = reranker.compute_score(pairs)

    for i, n in enumerate(nodes):
        n.score = scores[i]

    nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
    return nodes[:top_k]
