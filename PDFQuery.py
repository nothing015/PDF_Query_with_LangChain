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

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate, LLMChain
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Provide the path of the PDF file
pdf_path = r"C:\Users\nnuai\Downloads\astronomy_textbook.pdf"
pdfreader = PdfReader(pdf_path)

# Read text from PDF
raw_text = ""
for index, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content is not None:
        raw_text += content

# Splitting the text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200
)
texts = text_splitter.split_text(raw_text)

# Use the first chunk or a summarized version of the text to create a message
first_chunk = texts[0] if texts else ""
summary_text = first_chunk[:1000]  # Truncate to first 1000 characters for simplicity

# Get user input for the custom prompt
user_prompt = input("Enter your prompt for the chat: ")

# Record the start time
start_time = time.time()

# Initialize the LLaMA model from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the correct model name on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a custom LLM class for Hugging Face models
class HuggingFaceLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create a HuggingFace LLM instance
llm = HuggingFaceLLM(model, tokenizer)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["content", "user_prompt"],
    template="Here is some content from a PDF: {content}. {user_prompt}"
)

# Create an LLM chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Generate a response based on the PDF content and user prompt
response = llm_chain.run({
    "content": summary_text,
    "user_prompt": user_prompt
})

# Record the end time
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

# Print the time taken
print(f"Time taken to get a response: {time_taken:.2f} seconds")

print("Chatbot Response:\n")
print(response)
