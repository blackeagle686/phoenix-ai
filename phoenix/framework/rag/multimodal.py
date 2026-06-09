import os
import uuid
import asyncio
import hashlib
import base64
import mimetypes
from typing import Optional, List, Dict, Any
from phoenix.framework.rag.base import BaseRAG
from phoenix.framework.rag.config import MultiModalRAGConfig
from phoenix.services.llm.base import BaseLLM, BaseVLM
from phoenix.services.vector.base import BaseVectorDB
from phoenix.services.vector.embeddings import BaseEmbeddings
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.MultiModalRAG")


class MultiModalRAG(BaseRAG):
    """MultiModal RAG that handles text, images, PDFs, and audio.

    Extends standard RAG with vision-language model integration.
    Images can be ingested (captioned and indexed) and queried against.
    Queries can include image attachments for visual QA.

    Usage:
        mmrag = MultiModalRAG(
            vlm=OpenAIVLM(),
            caption_images=True,
        )
        await mmrag.ingest("/path/to/docs")
        await mmrag.ingest_images("/path/to/images")
        answer = await mmrag.query("Describe the architecture diagram")
        answer = await mmrag.query_with_image("What is in this image?", "/path/to/img.png")
    """

    def __init__(
        self,
        config: MultiModalRAGConfig = None,
        llm: BaseLLM = None,
        vlm: BaseVLM = None,
        vector_db: BaseVectorDB = None,
        embeddings: BaseEmbeddings = None,
        cache=None,
        **kwargs
    ):
        cfg = config or MultiModalRAGConfig()
        super().__init__(
            config=cfg,
            llm=llm,
            vector_db=vector_db,
            embeddings=embeddings,
            cache=cache,
            **kwargs
        )
        self.vlm = vlm
        self._vlm_initialized = False

    async def _ensure_vlm(self):
        if self.vlm and not self._vlm_initialized:
            if hasattr(self.vlm, "client") and self.vlm.client is None:
                if hasattr(self.vlm, "init"):
                    await self.vlm.init()
            self._vlm_initialized = True

    def _is_image(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

    def _is_audio(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

    def _preprocess_image(self, image_path: str) -> str:
        try:
            from PIL import Image
            import tempfile
            img = Image.open(image_path)
            w, h = img.size
            max_size = self.config.image_max_size
            if w > max_size or h > max_size:
                if w > h:
                    nw, nh = max_size, int(max_size * h / w)
                else:
                    nh, nw = max_size, int(max_size * w / h)
                img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.convert("RGB").save(tmp.name, "JPEG", quality=self.config.image_quality, optimize=True)
            return tmp.name
        except Exception:
            return image_path

    async def _caption_image(self, image_path: str) -> str:
        if not self.vlm:
            return f"[Image: {os.path.basename(image_path)}]"
        await self._ensure_vlm()
        processed = self._preprocess_image(image_path)
        try:
            caption = await self.vlm.generate_with_image(
                "Describe this image in detail. Include all visible text, objects, and layout.",
                processed
            )
            return caption
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return f"[Image: {os.path.basename(image_path)}]"
        finally:
            if processed != image_path and os.path.exists(processed):
                try:
                    os.remove(processed)
                except Exception:
                    pass

    async def _transcribe_audio(self, audio_path: str) -> str:
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "")
        except ImportError:
            logger.warning("whisper not installed. Audio transcription skipped.")
            return f"[Audio: {os.path.basename(audio_path)}]"
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return f"[Audio: {os.path.basename(audio_path)}]"

    async def ingest_images(self, path: str):
        """Ingest images by captioning them and indexing the captions."""
        await self._ensure_init()
        await self._ensure_vlm()

        files = []
        if os.path.isfile(path) and self._is_image(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, fnames in os.walk(path):
                for f in fnames:
                    fp = os.path.join(root, f)
                    if self._is_image(fp):
                        files.append(fp)

        if not files:
            logger.info("No image files found.")
            return

        sem = asyncio.Semaphore(5)
        chunks = []
        meta = []

        async def _process(img_path):
            async with sem:
                caption = await self._caption_image(img_path)
                return img_path, caption

        logger.info(f"Captioning {len(files)} images...")
        results = await asyncio.gather(*[_process(f) for f in files])

        for img_path, caption in results:
            if caption:
                chunks.append(caption)
                meta.append({
                    "source": os.path.basename(img_path),
                    "path": img_path,
                    "media_type": "image",
                    "is_parent": False
                })

        if chunks:
            await self.vector_db.add(texts=chunks, metadatas=meta)
            logger.info(f"Indexed {len(chunks)} image captions.")

    async def ingest_audio(self, path: str):
        """Ingest audio files by transcribing and indexing."""
        await self._ensure_init()

        files = []
        if os.path.isfile(path) and self._is_audio(path):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, fnames in os.walk(path):
                for f in fnames:
                    fp = os.path.join(root, f)
                    if self._is_audio(fp):
                        files.append(fp)

        if not files:
            logger.info("No audio files found.")
            return

        for audio_path in files:
            logger.info(f"Transcribing {audio_path}...")
            transcript = await self._transcribe_audio(audio_path)
            if transcript and not transcript.startswith("[Audio:"):
                chunks = self._chunk_text(transcript, self.config.chunk_size, self.config.chunk_overlap)
                metas = [{
                    "source": os.path.basename(audio_path),
                    "path": audio_path,
                    "media_type": "audio",
                    "is_parent": False
                } for _ in chunks]
                await self.vector_db.add(texts=chunks, metadatas=metas)
                logger.info(f"Indexed {len(chunks)} audio chunks from {audio_path}.")

    async def ingest_multimodal(self, path: str):
        """Ingest a directory containing mixed media (text, images, audio)."""
        await self._ensure_init()
        text_files = []
        image_files = []
        audio_files = []

        if os.path.isfile(path):
            if self._is_image(path):
                image_files.append(path)
            elif self._is_audio(path):
                audio_files.append(path)
            else:
                text_files.append(path)
        elif os.path.isdir(path):
            for root, _, fnames in os.walk(path):
                if any(p.startswith('.') for p in root.split(os.sep)):
                    continue
                for f in fnames:
                    fp = os.path.join(root, f)
                    if self._is_image(fp):
                        image_files.append(fp)
                    elif self._is_audio(fp):
                        audio_files.append(fp)
                    else:
                        text_files.append(fp)

        tasks = []
        if text_files:
            tasks.append(self.ingest(path))
        if image_files:
            tasks.append(self.ingest_images(path))
        if audio_files:
            tasks.append(self.ingest_audio(path))
        if tasks:
            await asyncio.gather(*tasks)

    async def query_with_image(self, question: str, image_path: str, system_prompt: str = None) -> str:
        """Query using both text retrieval and an image attachment."""
        await self._ensure_init()
        await self._ensure_vlm()

        docs = await self.retrieve(question)
        context = "\n\n".join(d.get("content", "")[:500] for d in docs[:3])

        sp = system_prompt or self.config.system_prompt or "You are a multimodal AI assistant."
        prompt = f"{sp}\n\nRetrieved Context:\n{context}\n\nUser Question: {question}"

        if self.vlm:
            processed = self._preprocess_image(image_path)
            try:
                return await self.vlm.generate_with_image(prompt, processed)
            except Exception as e:
                logger.error(f"VLM query failed: {e}. Falling back to text-only.")
            finally:
                if processed != image_path and os.path.exists(processed):
                    try:
                        os.remove(processed)
                    except Exception:
                        pass

        return await self.llm.generate(prompt, max_tokens=self.config.max_output_tokens)

    async def query(self, question: str, system_prompt: str = None, history: str = None) -> str:
        return await super().query(question, system_prompt=system_prompt, history=history)
