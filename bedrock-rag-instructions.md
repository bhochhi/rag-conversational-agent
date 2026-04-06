# Copilot Instructions: Build a RAG Search App with AWS Bedrock

## Project Goal

Build a simple Retrieval-Augmented Generation (RAG) web app that:
- Indexes PDF documents into a Chroma vector store using **AWS Bedrock Titan embeddings**
- Answers user questions by retrieving relevant document chunks and sending them to an **AWS Bedrock LLM** (Amazon Titan Text)
- Exposes a minimal Flask web UI for querying

No Ollama. No OpenAI. No Anthropic. AWS Bedrock only.

---

## Tech Stack

| Concern         | Technology                                      |
|-----------------|-------------------------------------------------|
| Embeddings      | `amazon.titan-embed-text-v2:0` via AWS Bedrock  |
| LLM             | `amazon.titan-text-express-v1` via AWS Bedrock  |
| Vector Store    | ChromaDB (local, persisted to disk)             |
| RAG Framework   | LangChain (`langchain-aws`, `langchain-chroma`) |
| Web Server      | Flask                                           |
| PDF Loading     | `pypdf` via LangChain                           |
| AWS SDK         | `boto3`                                         |
| Config          | `python-dotenv`                                 |

---

## Project Structure

Create the project with exactly this layout:

```
rag-bedrock/
├── .env                        # AWS credentials and app config
├── requirements.txt
├── populate_database.py        # Script: load PDFs → embed → store in Chroma
├── app.py                      # Flask web app: query endpoint + UI
├── embeddings/
│   └── bedrock_embeddings.py   # Wraps BedrockEmbeddings from langchain_aws
├── retrieval/
│   └── rag_retriever.py        # Queries Chroma DB, formats results
├── llm/
│   └── bedrock_llm.py          # Wraps Titan Text via boto3 for generation
├── templates/
│   └── index.html              # Simple chat UI
├── data/                       # Place PDF files here (not committed)
└── chroma_db/                  # Chroma persisted store (auto-created, not committed)
```

---

## Environment Configuration

### `.env`

```dotenv
# AWS credentials (or use an IAM role / AWS profile instead)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# App settings
DATA_PATH=data/
VECTOR_DB_PATH=chroma_db/
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
LLM_MODEL_ID=amazon.titan-text-express-v1
NUM_RELEVANT_DOCS=4
```

> Users must have Bedrock model access enabled in their AWS account for both model IDs above.
> Grant access at: AWS Console → Bedrock → Model access.

---

## Implementation Specifications

### `requirements.txt`

```
flask
python-dotenv
boto3
langchain==1.1.3
langchain-core==1.2.5
langchain-community==0.4.1
langchain-aws==0.2.0
langchain-chroma==1.0.0
langchain-text-splitters==1.0.0
langchain-huggingface
pypdf
chromadb
pytest
```

---

### `embeddings/bedrock_embeddings.py`

```python
import boto3
from langchain_aws import BedrockEmbeddings


def get_embedding_function(model_id: str, region: str) -> BedrockEmbeddings:
    """Return a LangChain-compatible embedding function backed by AWS Bedrock Titan."""
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return BedrockEmbeddings(client=bedrock_client, model_id=model_id)
```

---

### `llm/bedrock_llm.py`

```python
import json
import boto3


PROMPT_TEMPLATE = """Based only on the following context:

{context}

---

Answer the following question: {question}
Go straight to the answer without mentioning that you are basing it on the provided context.
"""


class BedrockTitanLLM:
    """Invokes amazon.titan-text-express-v1 via boto3 for text generation."""

    def __init__(self, model_id: str, region: str):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def generate_response(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9,
            }
        })

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"].strip()
```

---

### `retrieval/rag_retriever.py`

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document


class RAGRetriever:
    def __init__(self, vector_db_path: str, embedding_function):
        self.db = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embedding_function,
        )

    def query(self, query_text: str, k: int = 4) -> list[tuple[Document, float]]:
        return self.db.similarity_search_with_score(query_text, k=k)

    def format_results(self, results: list[tuple[Document, float]]) -> tuple[str, list[str]]:
        context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        sources = list({self._format_source(doc.metadata) for doc, _ in results})
        return context, sources

    def _format_source(self, metadata: dict) -> str:
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        filename = source.replace("\\", "/").split("/")[-1]
        return f"{filename} (page {page})"
```

---

### `populate_database.py`

```python
import os
import argparse
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings.bedrock_embeddings import get_embedding_function

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db/")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return splitter.split_documents(documents)


def assign_chunk_ids(chunks):
    """Assign stable IDs like 'source:page:chunk_index' to avoid duplicates."""
    last_page_id = None
    chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        page_id = f"{source}:{page}"
        if page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0
        last_page_id = page_id
        chunk.metadata["id"] = f"{page_id}:{chunk_index}"
    return chunks


def add_to_chroma(chunks, db: Chroma):
    chunks = assign_chunk_ids(chunks)
    existing_ids = set(db.get()["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to the database.")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    else:
        print("No new documents to add.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear the vector DB before indexing.")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
            print(f"Cleared database at {VECTOR_DB_PATH}")

    embedding_function = get_embedding_function(EMBEDDING_MODEL_ID, AWS_REGION)
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks, db)


if __name__ == "__main__":
    main()
```

---

### `app.py`

```python
import os
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from embeddings.bedrock_embeddings import get_embedding_function
from retrieval.rag_retriever import RAGRetriever
from llm.bedrock_llm import BedrockTitanLLM

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "chroma_db/")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "amazon.titan-text-express-v1")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
NUM_RELEVANT_DOCS = int(os.getenv("NUM_RELEVANT_DOCS", "4"))

app = Flask(__name__)

embedding_function = get_embedding_function(EMBEDDING_MODEL_ID, AWS_REGION)
retriever = RAGRetriever(vector_db_path=VECTOR_DB_PATH, embedding_function=embedding_function)
llm = BedrockTitanLLM(model_id=LLM_MODEL_ID, region=AWS_REGION)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    query_text = request.json.get("query_text", "").strip()
    if not query_text:
        return jsonify(error="Query text is required."), 400

    results = retriever.query(query_text, k=NUM_RELEVANT_DOCS)
    context, sources = retriever.format_results(results)
    answer = llm.generate_response(context=context, question=query_text)

    sources_html = "<br>".join(sources)
    response_text = f"{answer}<br><br><strong>Sources:</strong><br>{sources_html}"
    return jsonify(response=response_text)


if __name__ == "__main__":
    app.run(debug=True)
```

---

### `templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RAG Search – AWS Bedrock</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    h1 { color: #232f3e; }
    #query-input { width: 100%; padding: 10px; font-size: 16px; box-sizing: border-box; }
    #submit-btn { margin-top: 10px; padding: 10px 24px; background: #ff9900; border: none; color: #fff; font-size: 16px; cursor: pointer; border-radius: 4px; }
    #submit-btn:hover { background: #e68a00; }
    #response-box { margin-top: 24px; padding: 16px; background: #f8f8f8; border-left: 4px solid #232f3e; white-space: pre-wrap; }
    #loading { display: none; color: #888; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>RAG Search</h1>
  <p>Powered by <strong>AWS Bedrock</strong> – Titan Embeddings + Titan Text</p>

  <textarea id="query-input" rows="3" placeholder="Ask a question about your documents..."></textarea>
  <br>
  <button id="submit-btn" onclick="submitQuery()">Ask</button>
  <div id="loading">Thinking...</div>
  <div id="response-box" style="display:none;"></div>

  <script>
    async function submitQuery() {
      const queryText = document.getElementById("query-input").value.trim();
      if (!queryText) return;

      document.getElementById("loading").style.display = "block";
      document.getElementById("response-box").style.display = "none";

      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_text: queryText }),
      });

      const data = await res.json();
      document.getElementById("loading").style.display = "none";
      const box = document.getElementById("response-box");
      box.innerHTML = data.response || data.error;
      box.style.display = "block";
    }

    document.getElementById("query-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitQuery(); }
    });
  </script>
</body>
</html>
```

---

## Setup and Run Instructions

Tell Copilot's agent to generate these exact shell commands in the README:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in your AWS credentials + region

# 4. Place your PDF files in the data/ folder

# 5. Index documents into Chroma (run once, or with --reset to rebuild)
python populate_database.py

# 6. Start the web app
python app.py
# Open http://localhost:5000
```

---

## Important Constraints for Copilot

- **Do not add Ollama, OpenAI, or Anthropic dependencies**. AWS Bedrock is the only external AI provider.
- **Do not add an admin settings page**. Keep the UI to a single search page.
- **Use `boto3` directly** in `BedrockTitanLLM` (not LangChain's ChatBedrock) to keep the Titan Text API call explicit and easy to understand.
- **Use `langchain_aws.BedrockEmbeddings`** for embeddings since LangChain handles the Titan embedding request format correctly.
- All AWS auth is handled by `boto3` default credential chain (env vars → `~/.aws/credentials` → IAM role). Do not hard-code credentials.
- The `chroma_db/` and `data/` directories should be listed in `.gitignore`.

---

## `.gitignore` (generate this file too)

```
venv/
__pycache__/
*.pyc
.env
chroma_db/
data/
```
