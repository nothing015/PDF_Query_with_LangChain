# Retrieval-Augmented Generation (RAG) Process Breakdown:

## Data Loading:

I begin by loading content from three URLs. I use the WebBaseLoader to fetch and parse these web pages into document format.

## Text Splitting:

Next, I split the loaded documents into manageable chunks using the CharacterTextSplitter. This is essential for preparing the content for embedding creation by breaking the text into smaller pieces.

## Embedding Creation:

To convert the document chunks into dense vectors (embeddings), I use the OllamaEmbeddings model along with the nomic-embed-text embedding model. These vectors are stored in the Chroma vectorstore. Storing the document embeddings is a critical part of the RAG process, as it allows me to perform similarity search during retrieval.

## Retrieval:

I use the Chroma vectorstore to retrieve relevant document chunks based on a similarity search when I query the system with questions like "What is Ollama?"

## Generation:

Before RAG: I generate a response using the Mistral model with the template What is {topic} without any retrieved context.

After RAG: I enhance the response by including the retrieved document chunks as context. This is where the "retrieval-augmented" part of RAG comes into play, using the template to generate an answer based on the provided context.

# To include a PDF file as part of the RAG process:

## Load the PDF Document:

I use the following code to extract the text from a PDF:
```
from PyPDF2 import PdfReader

pdf_path = r"path_to_your_pdf.pdf"
pdfreader = PdfReader(pdf_path)

raw_text = ""
for page in pdfreader.pages:
    raw_text += page.extract_text()

```
## Convert the PDF Text to Embeddings and Add to the VectorStore:

After extracting the text from the PDF, I use the same text_splitter and Chroma embedding approach to convert it into chunks and add it to the vectorstore:
```
doc_splits = text_splitter.split_text(raw_text)

vectorstore.add_documents(doc_splits)  # Add PDF chunks to the vectorstore

```
