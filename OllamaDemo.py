# Required Packages
# pip install PyPDF2
# pip install ollama
from PyPDF2 import PdfReader
import ollama
import time  # Import the time module

# Provide the path of PDF file
pdf_path = r"C:\Users\nnuai\Downloads\astronomy_textbook.pdf"
pdfreader = PdfReader(pdf_path)

# Read text from PDF
raw_text = ""
for index, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content is not None:
        raw_text += content

# Splitting the text into manageable chunks
class CharacterTextSplitter:
    def __init__(self, separator="\n", chunk_size=800, chunk_overlap=200, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text):
        splits = []
        current_chunk = []
        current_length = 0

        for line in text.split(self.separator):
            current_chunk.append(line)
            current_length += self.length_function(line)

            if current_length >= self.chunk_size:
                splits.append(self.separator.join(current_chunk))
                current_chunk = current_chunk[-self.chunk_overlap:]  # retain overlap
                current_length = sum(self.length_function(x) for x in current_chunk)

        if current_chunk:
            splits.append(self.separator.join(current_chunk))

        return splits

# Create an instance of CharacterTextSplitter
text_splitter = CharacterTextSplitter()

# Split the extracted text into chunks
texts = text_splitter.split_text(raw_text)

# Use the first chunk or a summarized version of the text to create a message for ollama.chat
# In this example, we'll just use the first chunk
first_chunk = texts[0] if texts else ""

# If first_chunk is too large, you might want to summarize it or truncate it
summary_text = first_chunk[:1000]  # Truncate to first 1000 characters for simplicity

# Get user input for the custom prompt
user_prompt = input("Enter your prompt for the chat: ")

# Record the start time
start_time = time.time()

# Use ollama to generate a response based on the PDF content and user prompt
response = ollama.chat(
    model="llama3",
    messages=[
        # {
        #     "role": "system",
        #     "content": "Act like a pirate, speaking like one and adding arrrgh at the end of each line.",
        # },
        {
            "role": "user",
            "content": f"Here is some content from a PDF: {texts}. {user_prompt}",
        },
    ],
)

# Record the end time
end_time = time.time()

# Calculate the time taken
time_taken = end_time - start_time

# Print the time taken
print(f"Time taken to get a response: {time_taken:.2f} seconds")

print("Chatbot Response:\n")
print(response["message"]["content"])
