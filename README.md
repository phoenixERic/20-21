COMPATABLE WITH PYTHON 3.11 

KG-RAG
├── PDF
├── src
     | __Chunker
     |__injest
     |__pdf_extractor
     |___query
     |__rag_engine
     |__vector_store
├── requirements.txt
├── .env
└── README.md


 Try with python 3.11 as the new version python 3.14 is not compatible and need extra tools to run in vs code */
## Setup (in VS Code)

### 1. Open the project folder
Open VS Code → `File > Open Folder` → select `pdf-rag-project`.

### 2. Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` `` or `View > Terminal`) and run:
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.
In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select
Interpreter" → choose the one inside `venv`.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
This installs: `pdfplumber` (PDF/table extraction), `sentence-transformers`
(local embeddings), `chromadb` (local vector database), `google-generativeai`
(Gemini API client), `tiktoken` (token counting), `python-dotenv`.

> First install may take a few minutes — `sentence-transformers` pulls in
> PyTorch. That's normal.

### 4. Get a free Gemini API key
Go to **https://aistudio.google.com/app/apikey**, sign in, click "Create API
key" — it's free for moderate usage.

### 5. Set up your API key
Copy `.env.example` to a new file named `.env` in the project root:
```bash
cp .env.example .env       # Mac/Linux
copy .env.example .env     # Windows
```
Open `.env` and paste your real key:
```
GEMINI_API_KEY=AIzaSy...your_actual_key...
```

### 6. Add your PDFs
Drop all your PDF files into `data/pdfs/`. You can add as many as you want —
the system handles multiple PDFs natively and always tracks which file each
answer came from.

### 7. Ingest your PDFs (run once, or whenever you add new PDFs)
```bash
cd src
python ingest.py
```
You'll see progress logs as it extracts pages, builds chunks, and embeds
everything. The first run also downloads the embedding model (~80MB) —
that's a one-time download.

### 8. Ask questions!
```bash
python query.py
```
Example session:
```
❓ Your question: What was Apple's total net sales in Q1 2023?

ANSWER:
Apple's total net sales for the three months ended December 31, 2022 were
$117,154 million.

📚 Sources:
   - 2023_Q1_AAPL.pdf, page 4 (similarity: 0.81)
```

/* unable to upload with the folders due to time constraint and just uploaded files */
