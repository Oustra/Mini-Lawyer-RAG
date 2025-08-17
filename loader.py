from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pickle

# --- Load French PDFs ---
loader_fr_pdf = DirectoryLoader(
    "data/fr",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader_fr_pdf.load()
print(f"Loaded {len(documents)} documents")

# --- Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")

# --- Multilingual Embeddings (optimized for RAG) ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Optional: save the raw splits for faster reruns
with open("embeddings/splits.pkl", "wb") as f:
    pickle.dump(splits, f)
print("Saved split documents to embeddings/splits.pkl")

# --- Embed chunks and store in Chroma with progress ---
print("Embedding and storing chunks in ChromaDB...")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
for i, chunk in enumerate(splits, 1):
    db.add_documents([chunk])  # Add one chunk at a time
    if i % 50 == 0 or i == len(splits):  # Print progress every 50 chunks
        print(f"Embedded & stored {i}/{len(splits)} chunks")

# Persist DB to disk
db.persist()
print("All documents embedded & stored in ChromaDB successfully!")
