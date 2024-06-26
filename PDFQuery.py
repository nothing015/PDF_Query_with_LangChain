#Have these packages installed
"""
openai: This is the official OpenAI Python client library,
used for accessing the OpenAI API. This library allows developers
to interact with OpenAI's language models, such as GPT-4,
to generate text, answer questions, and perform other natural
language processing tasks.

PyPDF2: PyPDF2 is a Python library that enables users to manipulate
PDF files. With this library, you can extract text from PDFs, merge
multiple PDFs into a single file, split a PDF into individual pages,
and perform various other tasks involving PDF documents.

faiss-cpu: FAISS (Facebook AI Similarity Search) is a library for
efficient similarity search and clustering of dense vectors. The
faiss-cpu version is tailored for CPU usage and helps in performing
operations like nearest neighbor search. It is widely used in
applications requiring efficient vector search, such as document
retrieval and recommendation systems.

tiktoken: This package is used for tokenizing text inputs according
to OpenAI's models. Tokenization is the process of converting text
into a sequence of tokens, which are the units that models process.
This library helps ensure that text is correctly tokenized in a way
that aligns with OpenAI's expectations, which is important for accurate
text generation and analysis.
"""
# pip install openai
# pip install PyPDF2
# pip install faiss-cpu
# pip install tiktoken

# This class allows for reading PDF files and extracting text from them
from PyPDF2 import PdfReader

# This class is used to generate embeddings (vector representations) of text using OpenAI's models
from langchain_openai import OpenAIEmbeddings

# This class is used to split text into smaller chunks based on character count,
# which is useful for processing long texts in manageable segments
from langchain.text_splitter import CharacterTextSplitter

# The FAISS class in langchain is used to create and query a vector store for embeddings
from langchain_community.vectorstores import FAISS


import os
# Here the OpenAI API key is entered. Mine is confidential hence I'm using an environment variable to hide my API
# Key inside my system
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Provide the path of PDF file
pdfreader = PdfReader(r"C:\Users\nnuai\Downloads\astronomy_textbook.pdf")

from typing_extensions import Concatenate
# read text from pdf
rawText = ""
for index, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content is not None:
        rawText += content

# print(rawText)
# Splitting the text using Character Text Split such that it should not increase token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1500,
    chunk_overlap = 200,
    length_function = len,
)

# Splitting text into manageable chunks
texts = text_splitter.split_text(rawText)

# Download embeddings from OpenAI for the text chunks
embeddings = OpenAIEmbeddings()

# This embeds the 'texts' into embeddings and makes it a langChain vectorstore
document_search = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import (load_qa_chain)
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "What is a planet?"
docs = document_search.similarity_search(query)
print(chain.run(input_documents=docs, question = query))