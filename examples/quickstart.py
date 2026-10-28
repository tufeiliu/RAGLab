from raglab import RAGLab, RAGConfig

lab = RAGLab(RAGConfig(db_dir="example_db", query_rewrite=True,
                       query_expansion=3, rerank=True))
lab.ingest_texts(["Acme Analytics was founded in 2022 and sells forecasting software."])
result = lab.ask("When was the company founded?")
print(result.answer)
print(result.trace)
