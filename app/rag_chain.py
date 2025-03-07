import os
from operator import itemgetter

import asyncio

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnableParallel
from typing_extensions import TypedDict

from config import EMBEDDING_MODEL, PG_COLLECTION_NAME

load_dotenv()

# print("""POSTGRES_URL = {os.getenv("POSTGRES_URL")}""")
print(os.getenv("POSTGRES_URL"))


vector_store = PGVector(
    collection_name=PG_COLLECTION_NAME,
    # connection_string="postgresql+psycopg://postgres@localhost:5432/pdf_rag_vectors",
    connection_string=os.getenv("POSTGRES_URL"),
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOllama(
    temperature=0,
    model=EMBEDDING_MODEL,
    streaming=True
)

class RagInput(TypedDict):
    question: str

final_chain = (
        RunnableParallel(
            context=itemgetter("question") | vector_store.as_retriever(),
            question=itemgetter("question")
        ) |
        RunnableParallel(
            answer=(ANSWER_PROMPT | llm),
            docs=itemgetter("context")
        )
        ).with_types(input_type=RagInput)

#  Demo in action. Uncomment the following code to see the output
# FINAL_CHAIN_INVOKE = final_chain.astream_log({
#     "question": "Why compilers can't easily transform operations into machine instructions rely on specific type?"
#     })
#
# async def __main__():
#     async for c in FINAL_CHAIN_INVOKE:
#         print(c)
#
# asyncio.run(__main__())
