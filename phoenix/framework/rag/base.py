import os
import uuid
import asyncio
import hashlib
import mmap
from typing import Optional, List, Dict, Any, Union
from abc import abstractmethod

from phoenix.framework.rag.config import RAGConfig
from phoenix.services.llm.openai import OpenAILLM
from phoenix.services.llm.base import BaseLLM
from phoenix.services.vector.chroma import ChromaVectorDB
from phoenix.services.vector.embeddings import SentenceTransformerEmbeddings, BaseEmbeddings
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.cache.redis_cache import RedisCache
from phoenix.services.cache.semantic import SemanticCache
from phoenix.services.retrieval.retriever import VectorRetriever
from phoenix.services.retrieval.composer import PromptComposer
from phoenix.services.retrieval.optimizer import Optimizer
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.RAG.Framework")


class BaseRAG:
    """Foundation class for all Phoenix RAG systems."""

    def __init__(
        self,
        config: RAGConfig = None,
        llm: BaseLLM = None,
        vector_db: BaseVectorDB = None,
        embeddings: BaseEmbeddings = None,
        cache=None,
        **kwargs
    ):
        self.config = config or RAGConfig()
        self._apply_kwargs(kwargs)

        self.embeddings = embeddings or SentenceTransformerEmbeddings(
            device=self.config.device
        )
        self.vector_db = vector_db or ChromaVectorDB(
            collection_name=self.config.collection_name,
            embedding_service=self.embeddings
        )
        self.llm = llm or OpenAILLM()
        self.cache = cache
        self.semantic_cache = SemanticCache(
            embeddings=self.embeddings,
            threshold=self.config.similarity_threshold
        )
        self.retriever = VectorRetriever(self.vector_db)
        self.composer = PromptComposer()
        self.optimizer = Optimizer()
        self._initialized = False

    def _apply_kwargs(self, kwargs: dict):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    async def init(self):
        if self._initialized:
            return
        if hasattr(self.vector_db, "init"):
            await self.vector_db.init()
        if hasattr(self.llm, "client") and self.llm.client is None:
            if hasattr(self.llm, "init"):
                await self.llm.init()
        if self.cache and hasattr(self.cache, "init"):
            await self.cache.init()
        self._initialized = True
        logger.info(f"{self.__class__.__name__} initialized.")

    async def _ensure_init(self):
        if not self._initialized:
            await self.init()

    # ---- INGESTION ----

    async def ingest(self, path: str):
        await self._ensure_init()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        valid_ext = (
            ".txt", ".md", ".pdf", ".docx", ".csv", ".json",
            ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h",
            ".java", ".go", ".rs", ".php", ".rb", ".sql", ".sh"
        )

        files = []
        if os.path.isfile(path):
            files.append(path)
        else:
            for root, _, fnames in os.walk(path):
                if any(p.startswith('.') for p in root.split(os.sep)):
                    continue
                for f in fnames:
                    if f.lower().endswith(valid_ext):
                        files.append(os.path.join(root, f))

        if not files:
            logger.info("No valid files found for ingestion.")
            return

        sem = asyncio.Semaphore(100)
        all_chunks = []
        all_meta = []

        async def _process(fpath):
            async with sem:
                def _work():
                    content = self._read_file(fpath)
                    if not content:
                        return None
                    chunks, meta = [], []
                    if self.config.parent_retrieval:
                        parents = self._chunk_text(content, self.config.parent_chunk_size, self.config.chunk_overlap)
                        for pc in parents:
                            pid = str(uuid.uuid4())
                            chunks.append(pc)
                            meta.append({"source": os.path.basename(fpath), "path": fpath, "is_parent": True, "doc_id": pid})
                            children = self._chunk_text(pc, self.config.child_chunk_size, self.config.chunk_overlap)
                            for cc in children:
                                chunks.append(cc)
                                meta.append({"source": os.path.basename(fpath), "path": fpath, "is_parent": False, "parent_id": pid})
                    else:
                        parts = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
                        chunks.extend(parts)
                        meta.extend([{"source": os.path.basename(fpath), "path": fpath, "is_parent": False} for _ in parts])
                    return chunks, meta
                try:
                    return await asyncio.to_thread(_work)
                except Exception as e:
                    logger.error(f"Error processing {fpath}: {e}")
                    return None

        logger.info(f"Ingesting {len(files)} files...")
        results = await asyncio.gather(*[_process(f) for f in files])
        for r in results:
            if r:
                c, m = r
                all_chunks.extend(c)
                all_meta.extend(m)

        if all_chunks:
            logger.info(f"Indexing {len(all_chunks)} chunks...")
            await self.vector_db.add(texts=all_chunks, metadatas=all_meta)
            logger.info("Indexing complete.")

    async def ingest_texts(self, texts: List[str], metadatas: List[dict] = None):
        await self._ensure_init()
        if not metadatas:
            metadatas = [{"source": "direct_input"} for _ in texts]
        all_chunks = []
        all_meta = []
        for i, text in enumerate(texts):
            parts = self._chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            all_chunks.extend(parts)
            all_meta.extend([metadatas[i] for _ in parts])
        if all_chunks:
            await self.vector_db.add(texts=all_chunks, metadatas=all_meta)

    async def ingest_url(self, url: str):
        await self._ensure_init()
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            content = " ".join(soup.get_text(separator=" ").split())
            chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
            if chunks:
                await self.vector_db.add(
                    texts=chunks,
                    metadatas=[{"source": url} for _ in chunks]
                )
        except Exception as e:
            logger.error(f"URL ingestion failed for {url}: {e}")
            raise

    async def ingest_github(self, repo_url: str, branch: str = "main"):
        await self._ensure_init()
        import tempfile
        import subprocess
        import shutil
        tmp = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, repo_url, tmp],
                capture_output=True, text=True
            )
            if result.returncode != 0 and branch == "main":
                shutil.rmtree(tmp)
                tmp = tempfile.mkdtemp()
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "-b", "master", repo_url, tmp],
                    capture_output=True, text=True
                )
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
            await self.ingest(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def ingest_sql(
        self,
        connection_string: str,
        query: str,
        text_columns: Union[str, List[str]] = None,
        metadata_columns: List[str] = None,
        table: str = None
    ):
        """Ingest rows from a SQL database.

        Args:
            connection_string: SQLAlchemy-compatible connection string.
                Examples: "sqlite:///data.db", "postgresql://user:pass@host/db",
                          "mysql+pymysql://user:pass@host/db"
            query: Raw SQL query to execute. Ignored if `table` is provided.
            text_columns: Column(s) whose content gets chunked and indexed.
                If None, all columns are concatenated as key=value text.
            metadata_columns: Columns to store as metadata alongside each chunk.
            table: Shorthand to ingest an entire table (SELECT * FROM table).
        """
        await self._ensure_init()
        try:
            from sqlalchemy import create_engine, text as sql_text
        except ImportError:
            raise ImportError("sqlalchemy is required for SQL ingestion. pip install sqlalchemy")

        if table and not query:
            query = f"SELECT * FROM {table}"

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(sql_text(query))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchall()]

        if not rows:
            logger.info("SQL query returned 0 rows.")
            return

        if isinstance(text_columns, str):
            text_columns = [text_columns]

        all_chunks = []
        all_meta = []

        for row in rows:
            if text_columns:
                content = "\n".join(str(row.get(col, "")) for col in text_columns if row.get(col))
            else:
                content = "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)

            if not content.strip():
                continue

            meta_base = {"source": "sql", "connection": connection_string.split("@")[-1] if "@" in connection_string else connection_string}
            if metadata_columns:
                for mc in metadata_columns:
                    if mc in row:
                        meta_base[mc] = str(row[mc])

            chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
            all_chunks.extend(chunks)
            all_meta.extend([{**meta_base, "is_parent": False} for _ in chunks])

        if all_chunks:
            await self.vector_db.add(texts=all_chunks, metadatas=all_meta)
            logger.info(f"SQL ingestion complete: {len(rows)} rows -> {len(all_chunks)} chunks.")

    async def ingest_api(
        self,
        url: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        body: Dict[str, Any] = None,
        data_path: str = None,
        text_field: str = None,
        pagination: Dict[str, Any] = None
    ):
        """Ingest data from a REST API endpoint.

        Args:
            url: API endpoint URL.
            method: HTTP method (GET or POST).
            headers: Request headers (e.g. Authorization).
            body: JSON body for POST requests.
            data_path: Dot-notation path to drill into the response JSON.
                Example: "results.items" drills response["results"]["items"]
            text_field: Specific field in each item to use as text content.
                If None, the entire item is serialized as text.
            pagination: Auto-pagination config. Dict with keys:
                "next_field": dot-path to the next page URL in the response
                "max_pages": maximum number of pages to fetch (default 10)
        """
        await self._ensure_init()
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for API ingestion. pip install httpx")

        max_pages = (pagination or {}).get("max_pages", 10) if pagination else 1
        next_field = (pagination or {}).get("next_field") if pagination else None
        current_url = url
        page = 0
        all_chunks = []
        all_meta = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            while current_url and page < max_pages:
                if method.upper() == "POST":
                    resp = await client.post(current_url, headers=headers, json=body)
                else:
                    resp = await client.get(current_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                items = data
                if data_path:
                    for part in data_path.split("."):
                        if isinstance(items, dict):
                            items = items.get(part, [])
                        elif isinstance(items, list) and part.isdigit():
                            items = items[int(part)]
                        else:
                            items = []
                            break

                if not isinstance(items, list):
                    items = [items]

                for item in items:
                    if text_field and isinstance(item, dict):
                        content = str(item.get(text_field, ""))
                    elif isinstance(item, dict):
                        content = "\n".join(f"{k}: {v}" for k, v in item.items() if v is not None)
                    else:
                        content = str(item)

                    if not content.strip():
                        continue

                    chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
                    all_chunks.extend(chunks)
                    all_meta.extend([{"source": url, "is_parent": False} for _ in chunks])

                if next_field and isinstance(data, dict):
                    next_url = data
                    for part in next_field.split("."):
                        next_url = next_url.get(part) if isinstance(next_url, dict) else None
                        if next_url is None:
                            break
                    current_url = next_url
                else:
                    current_url = None
                page += 1

        if all_chunks:
            await self.vector_db.add(texts=all_chunks, metadatas=all_meta)
            logger.info(f"API ingestion complete: {len(all_chunks)} chunks from {url}.")

    async def ingest_google_drive(
        self,
        folder_id: str = None,
        file_ids: List[str] = None,
        credentials_path: str = None,
        credentials_json: Dict = None,
        mime_filter: List[str] = None
    ):
        """Ingest files from Google Drive.

        Requires google-api-python-client and google-auth.
        pip install google-api-python-client google-auth-oauthlib

        Args:
            folder_id: Google Drive folder ID to ingest all files from.
            file_ids: Specific file IDs to ingest.
            credentials_path: Path to service account JSON key file.
            credentials_json: Service account credentials as a dict (alternative to file).
            mime_filter: List of MIME types to include.
                Defaults to text, PDF, docs, sheets, CSV.
        """
        await self._ensure_init()
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "Google API client is required. "
                "pip install google-api-python-client google-auth-oauthlib"
            )

        import tempfile
        import io

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        if credentials_json:
            creds = service_account.Credentials.from_service_account_info(credentials_json, scopes=scopes)
        elif credentials_path:
            creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
        else:
            raise ValueError("Provide credentials_path or credentials_json for Google Drive access.")

        service = build("drive", "v3", credentials=creds)

        default_mimes = [
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
            "application/json",
            "text/markdown",
            "application/vnd.google-apps.document",
            "application/vnd.google-apps.spreadsheet",
        ]
        allowed_mimes = mime_filter or default_mimes

        target_files = []

        if file_ids:
            for fid in file_ids:
                meta = service.files().get(fileId=fid, fields="id,name,mimeType").execute()
                target_files.append(meta)

        if folder_id:
            query = f"'{folder_id}' in parents and trashed=false"
            page_token = None
            while True:
                resp = service.files().list(
                    q=query, fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token, pageSize=100
                ).execute()
                for f in resp.get("files", []):
                    if f["mimeType"] in allowed_mimes or not mime_filter:
                        target_files.append(f)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

        if not target_files:
            logger.info("No files found in Google Drive.")
            return

        google_export_map = {
            "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
            "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
        }

        tmp_dir = tempfile.mkdtemp()
        downloaded = []

        try:
            for f in target_files:
                fid = f["id"]
                fname = f["name"]
                mime = f["mimeType"]

                if mime in google_export_map:
                    export_mime, ext = google_export_map[mime]
                    content = service.files().export(fileId=fid, mimeType=export_mime).execute()
                    local_path = os.path.join(tmp_dir, fname + ext)
                    with open(local_path, "wb") as out:
                        out.write(content)
                else:
                    content = service.files().get_media(fileId=fid).execute()
                    ext = os.path.splitext(fname)[1] or ".txt"
                    local_path = os.path.join(tmp_dir, fname if ext else fname + ".txt")
                    with open(local_path, "wb") as out:
                        out.write(content)

                downloaded.append(local_path)

            if downloaded:
                logger.info(f"Downloaded {len(downloaded)} files from Google Drive. Ingesting...")
                await self.ingest(tmp_dir)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def ingest_urls(self, urls: List[str]):
        """Batch ingest multiple URLs concurrently."""
        await self._ensure_init()
        sem = asyncio.Semaphore(10)

        async def _do(url):
            async with sem:
                try:
                    await self.ingest_url(url)
                except Exception as e:
                    logger.error(f"Failed to ingest URL {url}: {e}")

        await asyncio.gather(*[_do(u) for u in urls])
        logger.info(f"Batch URL ingestion complete: {len(urls)} URLs.")

    async def ingest_multi(self, sources: List[Dict[str, Any]]):
        """Unified multi-source ingestion from a declarative source list.

        Each source dict must have a "type" key. Supported types:
            file, folder, url, urls, github, sql, api, google_drive, texts

        Example:
            await rag.ingest_multi([
                {"type": "folder", "path": "/data/docs"},
                {"type": "github", "repo_url": "https://github.com/user/repo"},
                {"type": "sql", "connection_string": "sqlite:///app.db", "query": "SELECT * FROM articles", "text_columns": "body"},
                {"type": "api", "url": "https://api.example.com/data", "data_path": "results", "text_field": "content"},
                {"type": "google_drive", "folder_id": "1ABC...", "credentials_path": "/keys/sa.json"},
                {"type": "urls", "urls": ["https://example.com/page1", "https://example.com/page2"]},
                {"type": "texts", "texts": ["raw text 1", "raw text 2"]},
            ])
        """
        await self._ensure_init()
        results = {"success": [], "failed": []}

        for src in sources:
            src_type = src.get("type", "")
            label = f"{src_type}:{src.get('path', src.get('url', src.get('repo_url', src.get('table', ''))))}"
            try:
                if src_type in ("file", "folder"):
                    await self.ingest(src["path"])

                elif src_type == "url":
                    await self.ingest_url(src["url"])

                elif src_type == "urls":
                    await self.ingest_urls(src["urls"])

                elif src_type == "github":
                    await self.ingest_github(
                        src["repo_url"],
                        branch=src.get("branch", "main")
                    )

                elif src_type == "sql":
                    await self.ingest_sql(
                        connection_string=src["connection_string"],
                        query=src.get("query", ""),
                        text_columns=src.get("text_columns"),
                        metadata_columns=src.get("metadata_columns"),
                        table=src.get("table")
                    )

                elif src_type == "api":
                    await self.ingest_api(
                        url=src["url"],
                        method=src.get("method", "GET"),
                        headers=src.get("headers"),
                        body=src.get("body"),
                        data_path=src.get("data_path"),
                        text_field=src.get("text_field"),
                        pagination=src.get("pagination")
                    )

                elif src_type == "google_drive":
                    await self.ingest_google_drive(
                        folder_id=src.get("folder_id"),
                        file_ids=src.get("file_ids"),
                        credentials_path=src.get("credentials_path"),
                        credentials_json=src.get("credentials_json"),
                        mime_filter=src.get("mime_filter")
                    )

                elif src_type == "texts":
                    await self.ingest_texts(
                        texts=src["texts"],
                        metadatas=src.get("metadatas")
                    )
                else:
                    logger.warning(f"Unknown source type: {src_type}")
                    results["failed"].append({"source": label, "error": f"Unknown type: {src_type}"})
                    continue

                results["success"].append(label)
                logger.info(f"Ingested source: {label}")

            except Exception as e:
                results["failed"].append({"source": label, "error": str(e)})
                logger.error(f"Failed to ingest {label}: {e}")

        logger.info(f"Multi-source ingestion done. Success: {len(results['success'])}, Failed: {len(results['failed'])}")
        return results

    # ---- RETRIEVAL ----

    async def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        await self._ensure_init()
        k = top_k or self.config.top_k
        optimized = self.optimizer.rewrite_query(query)

        if not self.config.fast_mode and self.config.query_expansion:
            queries = await self.optimizer.expand_query(optimized, llm=self.llm)
        else:
            queries = [optimized]

        all_docs = []
        seen = set()
        for q in queries:
            docs = await self.retriever.retrieve(q, hybrid=self.config.hybrid_search)
            for d in docs:
                c = d.get("content", "")
                if c not in seen:
                    all_docs.append(d)
                    seen.add(c)

        if all_docs and self.config.reranking:
            all_docs = self.optimizer.rerank(all_docs, optimized)
        if all_docs and not self.config.fast_mode and self.config.context_compression:
            all_docs = self.optimizer.compress_context(all_docs, optimized)
        return all_docs[:k]

    # ---- QUERY ----

    async def query(self, question: str, system_prompt: str = None, history: str = None) -> str:
        await self._ensure_init()
        optimized = self.optimizer.rewrite_query(question)

        if self.semantic_cache:
            hit = await self.semantic_cache.get_similar(optimized)
            if hit:
                logger.info("Semantic cache hit.")
                return hit

        if self.cache:
            key = f"rag:{hashlib.md5(optimized.encode()).hexdigest()}"
            cached = await self.cache.get(key)
            if cached:
                logger.info("Redis cache hit.")
                return cached

        search_query = optimized
        if not self.config.fast_mode and self.config.hyde_enabled:
            search_query = await self.optimizer.get_hyde_query(optimized, llm=self.llm)

        docs = await self.retrieve(search_query)
        prompt = self.composer.build_prompt(
            question, docs,
            system_prompt=system_prompt or self.config.system_prompt,
            history=history
        )
        response = await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

        if self.semantic_cache:
            await self.semantic_cache.add(optimized, response)
        if self.cache:
            await self.cache.set(key, response, ttl=self.config.cache_ttl)

        return response

    async def clear(self):
        await self._ensure_init()
        await self.vector_db.clear()
        logger.info("Vector DB cleared.")

    # ---- FILE READING ----

    def _read_file(self, path: str) -> str:
        if path.endswith(".pdf"):
            return self._read_pdf(path)
        if path.endswith(".docx"):
            return self._read_docx(path)
        if path.endswith(".csv"):
            return self._read_csv(path)
        if path.endswith(".json"):
            return self._read_json(path)
        try:
            size = os.path.getsize(path)
            if size == 0:
                return ""
            with open(path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    return m.read().decode("utf-8", errors="ignore")
        except Exception:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _read_pdf(self, path: str) -> str:
        for loader in [
            lambda: __import__("pypdf").PdfReader,
            lambda: __import__("fitz"),
            lambda: __import__("pdfplumber"),
        ]:
            try:
                mod = loader()
                if hasattr(mod, "pages"):
                    reader = mod(path) if callable(mod) else __import__("pypdf").PdfReader(path)
                    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
                if hasattr(mod, "open"):
                    doc = mod.open(path)
                    return "\n".join(p.get_text() for p in doc)
                with mod.open(path) as pdf:
                    return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
            except (ImportError, Exception):
                continue
        return ""

    def _read_docx(self, path: str) -> str:
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except (ImportError, Exception):
            return ""

    def _read_csv(self, path: str) -> str:
        try:
            import csv
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return "\n".join(",".join(row) for row in csv.reader(f))
        except Exception:
            return ""

    def _read_json(self, path: str) -> str:
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
        except Exception:
            return ""

    # ---- CHUNKING ----

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        separators = ["\n\n", "\n", ". ", " ", ""]

        def _split(t, seps):
            if len(t) <= size:
                return [t]
            if not seps:
                return [t[i:i+size] for i in range(0, len(t), size - overlap)]
            sep = seps[0]
            if sep == "":
                return [t[i:i+size] for i in range(0, len(t), size - overlap)]
            parts = t.split(sep)
            chunks = []
            current = ""
            for part in parts:
                part_s = part + (sep if part != parts[-1] else "")
                if len(current) + len(part_s) <= size:
                    current += part_s
                else:
                    if current:
                        chunks.append(current)
                    if len(part_s) > size:
                        chunks.extend(_split(part_s, seps[1:]))
                        current = ""
                    else:
                        current = part_s
            if current:
                chunks.append(current)
            return chunks

        return _split(text, separators)
