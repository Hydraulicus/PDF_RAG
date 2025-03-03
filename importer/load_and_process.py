import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

from config import EMBEDDING_MODEL, PG_COLLECTION_NAME

loader = DirectoryLoader(
    os.path.abspath("../source_docs"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
    sample_size=1,
)

docs = loader.load()

load_dotenv()
#
# if not os.getenv("OPENAI_API_KEY"):
#     print('The OPENAI_API_KEY is absent')
# else:
#     print('The OPENAI_API_KEY is present')
# OpenAIEmbeddings(
#     model=EMBEDDING_MODEL
# )
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
)

text_splitter = SemanticChunker(
    embeddings=embeddings
)


chunks = text_splitter.split_documents(docs)
# print(chunks)


# docker run --name pgvector -e POSTGRES_HOST_AUTH_METHOD=trust -p 5432:5432 -d ankane/pgvector

# connect to postgres
# psql "postgresql://postgres:postgres@localhost:5432"

# connect and create database
# psql "postgresql://postgres:postgres@localhost:5432" -c "create database pdf_rag_vectors"

# create
#  \c pdf_rag_vectors

#  \l to list databases

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=PG_COLLECTION_NAME,
    # connection_string=os.getenv("POSTGRES_URL"),
    connection_string="postgresql+psycopg://postgres@localhost:5432/pdf_rag_vectors",
    pre_delete_collection=True,

)