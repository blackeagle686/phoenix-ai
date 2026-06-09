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
