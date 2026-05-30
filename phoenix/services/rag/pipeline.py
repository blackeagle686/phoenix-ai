import os
from typing import List, Optional
from phoenix.services.retrieval.engine import InsightEngine
from phoenix.core.container import container
from phoenix.core.config import config

class RAGPipeline:
    def __init__(self, vector_db, primary, fallback=None, cache=None, semantic_cache=None, rag_config=None):
        self.vector_db = vector_db
        self.primary = primary
        self.fallback = fallback or primary
        self.engine = InsightEngine(vector_db, self.primary, self.fallback, cache, semantic_cache=semantic_cache, rag_config=rag_config)

    async def ingest(self, path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Loads documents from path, chunks them concurrently, and stores in vector DB.
        """
        print(f"[*] Starting Ingestion for path: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")

        # Common code and text extensions for ingestion
        valid_extensions = (
            ".txt", ".md", ".pdf", ".docx", ".csv", ".json", # Docs
            ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h", # Code
            ".java", ".go", ".rs", ".php", ".rb", ".sql", ".sh" # More Code
        )

        documents = []
        if os.path.isfile(path):
            documents.append(path)
        else:
            for root, _, files in os.walk(path):
                # Skip hidden directories (like .git)
                if any(part.startswith('.') for part in root.split(os.sep)):
                    continue
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        documents.append(os.path.join(root, file))

        all_chunks = []
        all_metadatas = []
        import asyncio
        import uuid

        # Bound concurrency to prevent OS file descriptor/memory exhaustion
        # and maximize sustained I/O throughput.
        semaphore = asyncio.Semaphore(100)

        async def process_doc(doc_path: str):
            async with semaphore:
                def _read_and_chunk():
                    content = self._read_file(doc_path)
                    if not content:
                        return None
                    
                    local_chunks = []
                    local_metadatas = []
                    
                    if config.RAG_PARENT_RETRIEVAL:
                        parents = self._chunk_text(content, config.RAG_PARENT_CHUNK_SIZE, config.RAG_CHUNK_OVERLAP)
                        for p_content in parents:
                            p_id = str(uuid.uuid4())
                            local_chunks.append(p_content)
                            local_metadatas.append({
                                "source": os.path.basename(doc_path), 
                                "path": doc_path, 
                                "is_parent": True, 
                                "doc_id": p_id
                            })
                            
                            children = self._chunk_text(p_content, config.RAG_CHILD_CHUNK_SIZE, config.RAG_CHUNK_OVERLAP)
                            for c_content in children:
                                local_chunks.append(c_content)
                                local_metadatas.append({
                                    "source": os.path.basename(doc_path), 
                                    "path": doc_path, 
                                    "is_parent": False, 
                                    "parent_id": p_id
                                })
                    else:
                        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                        local_chunks.extend(chunks)
                        local_metadatas.extend([{"source": os.path.basename(doc_path), "path": doc_path, "is_parent": False} for _ in chunks])
                    
                    return local_chunks, local_metadatas

                try:
                    return await asyncio.to_thread(_read_and_chunk)
                except Exception as e:
                    print(f"[!] Error processing {doc_path}: {e}")
                    return None

        print(f"[*] Processing and chunking {len(documents)} files concurrently...")
        results = await asyncio.gather(*(process_doc(doc) for doc in documents))

        for res in results:
            if res:
                c, m = res
                all_chunks.extend(c)
                all_metadatas.extend(m)

        if all_chunks:
            print(f"[*] Indexing {len(all_chunks)} units into Vector DB...")
            await self.vector_db.add(texts=all_chunks, metadatas=all_metadatas)
            print("[+] Indexing complete.")

    async def ingest_github(self, repo_url: str, branch: str = "main", chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Clones a GitHub repository and ingests its contents.
        """
        import tempfile
        import subprocess
        import shutil

        print(f"[*] cloning GitHub Repo: {repo_url} (branch: {branch})...")
        temp_dir = tempfile.mkdtemp()
        try:
            # Clone with depth 1 for speed
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, repo_url, temp_dir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                # If 'main' fails, try 'master' automatically
                if branch == "main" and "Remote branch main not found" in result.stderr:
                    print("[!] 'main' branch not found. Retrying with 'master'...")
                    shutil.rmtree(temp_dir)
                    temp_dir = tempfile.mkdtemp()
                    result = subprocess.run(
                        ["git", "clone", "--depth", "1", "-b", "master", repo_url, temp_dir],
                        capture_output=True, text=True
                    )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Git clone failed: {result.stderr}")

            print(f"[+] Repository cloned. Starting ingestion...")
            await self.ingest(temp_dir, chunk_size, chunk_overlap)
            print(f"[+] GitHub Ingestion successful: {repo_url}")

        finally:
            shutil.rmtree(temp_dir)
            print("[*] Temporary repository files cleaned up.")

    async def ingest_url(self, url: str, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Scrapes a URL, chunks the content, and stores in vector DB.
        """
        print(f"[*] Scraping URL: {url}...")
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            content = soup.get_text(separator=' ')
            content = " ".join(content.split())
            
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
            print(f"[+] Scraped and split into {len(chunks)} chunks.")
            
            if chunks:
                await self.vector_db.add(
                    texts=chunks, 
                    metadatas=[{"source": url} for _ in chunks]
                )
                print(f"[+] Indexed {url} successfully.")
        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    async def ingest_sql(self, connection_string: str, query: str, text_column: str, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Ingests data from a SQL database.
        """
        print(f"[*] Ingesting from SQL: {connection_string}...")
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.mappings().all()
            
            all_chunks = []
            all_metadatas = []
            
            for row in rows:
                content = str(row.get(text_column, ""))
                if not content: continue
                
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                metadata = {k: v for k, v in row.items() if k != text_column}
                metadata["source"] = "sql_query"
                all_metadatas.extend([metadata for _ in chunks])
            
            if all_chunks:
                await self.vector_db.add(texts=all_chunks, metadatas=all_metadatas)
                print(f"[+] Indexed {len(rows)} SQL rows successfully.")
        except Exception as e:
            print(f"[!] SQL Ingestion Error: {e}")

    async def ingest_api(self, url: str, method: str = "GET", headers: dict = None, data_path: str = None, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Ingests data from an external JSON API.
        """
        print(f"[*] Ingesting from API: {url}...")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    resp = await client.get(url, headers=headers)
                else:
                    resp = await client.post(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # If a data_path is provided (e.g., "results.items"), drill down
            items = data
            if data_path:
                for part in data_path.split('.'):
                    items = items.get(part, [])
            
            if not isinstance(items, list):
                items = [items]

            all_chunks = []
            all_metadatas = []
            for item in items:
                content = str(item) # Simplified: index the whole item as string if not specified better
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                all_metadatas.extend([{"source": url} for _ in chunks])

            if all_chunks:
                await self.vector_db.add(texts=all_chunks, metadatas=all_metadatas)
                print(f"[+] Indexed API response successfully.")
        except Exception as e:
            print(f"[!] API Ingestion Error: {e}")

    def _read_file(self, path: str) -> str:
        if path.endswith(".pdf"):
            try:
                # 1. Try pypdf (modern)
                from pypdf import PdfReader
                reader = PdfReader(path)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                pass
            
            try:
                # 2. Try fitz (PyMuPDF - very fast and common in Colab)
                import fitz
                doc = fitz.open(path)
                return "\n".join([page.get_text() for page in doc])
            except ImportError:
                pass

            try:
                # 3. Try pdfplumber
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            except ImportError:
                pass
                
            try:
                # 4. Try legacy PyPDF2
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except ImportError:
                print(f"[!] Warning: No PDF library installed (tried pypdf, fitz, pdfplumber, PyPDF2). Please `pip install pypdf` or `pymupdf` to read: {path}")
                return ""
            except Exception as e:
                print(f"[!] Error reading PDF {path}: {e}")
                return ""
        
        elif path.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                print(f"[!] Warning: python-docx not installed. Cannot read DOCX: {path}")
                return ""
        
        elif path.endswith((".xlsx", ".xls")):
            try:
                import pandas as pd
                df = pd.read_excel(path)
                return df.to_string()
            except ImportError:
                print(f"[!] Warning: pandas/openpyxl not installed. Cannot read Excel: {path}")
                return ""
            except Exception as e:
                print(f"[!] Error reading Excel {path}: {e}")
                return ""

        elif path.endswith(".csv"):
            try:
                import csv
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    return "\n".join([",".join(row) for row in reader])
            except Exception as e:
                print(f"[!] Error reading CSV {path}: {e}")
                return ""
        
        elif path.endswith(".json"):
            try:
                import json
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            except Exception as e:
                print(f"[!] Error reading JSON {path}: {e}")
                return ""
        
        import mmap
        try:
            if os.path.getsize(path) == 0:
                return ""
            with open(path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    return m.read().decode('utf-8', errors='ignore')
        except Exception:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        """
        Recursive character splitting to keep semantic units together.
        Splits by paragraphs, then sentences, then words.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_recursive(text: str, separators: List[str]) -> List[str]:
            if len(text) <= size:
                return [text]
            
            if not separators:
                return [text[i:i+size] for i in range(0, len(text), size - overlap)]
            
            sep = separators[0]
            if sep == "":
                # Character-level split fallback
                return [text[i:i+size] for i in range(0, len(text), size - overlap)]
            
            parts = text.split(sep)

            
            chunks = []
            current_chunk = ""
            
            for part in parts:
                # Add separator back except for the last part if it wasn't at the end
                part_with_sep = part + (sep if part != parts[-1] else "")
                
                if len(current_chunk) + len(part_with_sep) <= size:
                    current_chunk += part_with_sep
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # If the part itself is too large, split it with the remaining separators
                    if len(part_with_sep) > size:
                        chunks.extend(split_recursive(part_with_sep, separators[1:]))
                        current_chunk = "" # Reset
                    else:
                        current_chunk = part_with_sep
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Post-process: ensure overlap if possible
            # (Simplified overlap for recursive split: just join small chunks if they fit)
            return chunks

        return split_recursive(text, separators)

    async def query(self, question: str, session_id: Optional[str] = None, system_prompt: str = None, history: str = None) -> str:
        """
        Queries the RAG pipeline using the InsightEngine.
        """
        return await self.engine.query(question, system_prompt=system_prompt, history=history)

    async def clear_data(self) -> None:
        """Clears all data from the vector database."""
        await self.vector_db.clear()

