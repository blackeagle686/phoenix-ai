import base64
import httpx
from phoenix.services.llm.base import BaseVLM
from phoenix.core.config import config
from phoenix.services.observability.tracing import tracer
from phoenix.core.container import container
from typing import Optional

class LocalVLM(BaseVLM):
    _model_cache = {}

    def __init__(self):
        self.model = config.LOCAL_VLM_MODEL or "moondream"
        self.base_url = "http://localhost:11434/api/generate"
        self.hf_model = None
        self.processor = None
        self.is_ollama = False

    def is_available(self) -> bool:
        return bool(self.model) and config.LOAD_LOCAL_VLM

    async def init(self):
        if not self.model:
            return

        if not config.LOAD_LOCAL_VLM:
            print(f"[!] LocalVLM.init skipped for {self.model} because LOAD_LOCAL_VLM is False.")
            return

        if self.model not in LocalVLM._model_cache:
            print(f"[*] Initializing Local VLM Model: {self.model}...")
            try:
                from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM
                processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
                
                # Check config architecture to dynamically load the right class if Auto breaks
                config_opt = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
                architectures = getattr(config_opt, "architectures", [])
                
                hf_model = None
                
                # Check for bitsandbytes
                quant_kwargs = {}
                try:
                    import bitsandbytes
                    quant_kwargs = {"load_in_4bit": True}
                    print("[*] bitsandbytes available. Enabling 4-bit quantization to prevent OOM...")
                except ImportError:
                    print("[-] bitsandbytes not found. Loading model in standard precision (may cause OOM on small GPUs).")

                # Explicit Qwen3/Qwen2 VL support
                if "Qwen3VLForConditionalGeneration" in architectures:
                    try:
                        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
                        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                    except ImportError:
                        try:
                            from transformers import AutoModelForVision2Seq
                            hf_model = AutoModelForVision2Seq.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                        except ImportError:
                            hf_model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                elif "Qwen2VLForConditionalGeneration" in architectures:
                    from transformers import Qwen2VLForConditionalGeneration
                    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                else:
                    try:
                        from transformers import AutoModelForVision2Seq
                        hf_model = AutoModelForVision2Seq.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                    except (ImportError, ValueError):
                        hf_model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype="auto", trust_remote_code=True, **quant_kwargs)
                
                LocalVLM._model_cache[self.model] = {
                    "hf_model": hf_model,
                    "processor": processor,
                    "type": "transformers"
                }
                print("[+] VLM Loaded into memory cache.")
            except ImportError as ie:
                print(f"[!] {ie}. Falling back to Ollama.")
                LocalVLM._model_cache[self.model] = {"type": "ollama"}
            except Exception as e:
                print(f"[!] Failed to load model locally: {e}. Falling back to Ollama.")
                LocalVLM._model_cache[self.model] = {"type": "ollama"}

        cached = LocalVLM._model_cache[self.model]
        if cached["type"] == "transformers":
            self.hf_model = cached["hf_model"]
            self.processor = cached["processor"]
        else:
            self.is_ollama = True
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.base_url.replace('/api/generate', '')}/api/tags")
                    if resp.status_code != 200:
                        print(f"Warning: Ollama not responding at {self.base_url}.")
            except Exception:
                print("Warning: Could not connect to local Ollama. Ensure it is running.")

    async def generate_with_image(self, prompt: str, image_path: str, session_id: Optional[str] = None) -> str:
        if not self.hf_model and not self.is_ollama:
            if not config.LOAD_LOCAL_VLM:
                raise RuntimeError(
                    f"Attempted to use LocalVLM ({self.model}) but LOAD_LOCAL_VLM is False. "
                    "Initialize Phoenix AI with vlm=True to enable it."
                )
            await self.init()

        # Handle Memory
        memory = None
        try:
            memory = container.get("memory")
        except KeyError:
            pass

        context_prefix = ""
        messages = []
        if session_id and memory:
            # Retrieve history and semantic context
            semantic_context = await memory.search_memory(session_id, prompt)
            if semantic_context:
                context_prefix = f"Context from previous interactions:\n{semantic_context}\n\n"
            
            history = await memory.history.get(session_id)
            for item in history:
                messages.append(item["content"])

        span_id = tracer.start_span("LocalVLM.generate_with_image", {"model": self.model, "engine": "ollama" if self.is_ollama else "transformers"})

        response = ""
        if self.is_ollama:
            try:
                # Ollama vision prompt construction
                ollama_prompt = ""
                for m in messages:
                    content = m["content"]
                    if isinstance(content, list):
                        # Extract text if complex
                        content = next((c["text"] for c in content if c["type"] == "text"), "")
                    ollama_prompt += f"{m['role'].capitalize()}: {content}\n"
                
                ollama_prompt += f"User: {context_prefix}{prompt}\nAssistant:"
                
                response = await self._ollama_generate(ollama_prompt, image_path)
                words = (len(ollama_prompt.split()) + len(response.split()))
                usage = {"total_tokens": int(words * 1.5)}
                tracer.end_span(span_id, status="success", usage=usage)
            except Exception as e:
                tracer.end_span(span_id, status="error", error=str(e))
                raise
        else:
            try:
                from PIL import Image
                import torch
                image = Image.open(image_path).convert("RGB")
                
                # Add current message
                messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{context_prefix}{prompt}"}
                    ]
                })

                if hasattr(self.processor, "apply_chat_template"):
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.hf_model.device)
                else:
                    # Fallback
                    inputs = self.processor(images=image, text=f"{context_prefix}{prompt}", return_tensors="pt").to(self.hf_model.device)

                with torch.no_grad():
                    generated_ids = self.hf_model.generate(**inputs, max_new_tokens=config.SECURITY_MAX_OUTPUT_LENGTH)
                    
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response = output[0]
                
                usage = {
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": len(generated_ids_trimmed[0]),
                    "total_tokens": inputs.input_ids.shape[1] + len(generated_ids_trimmed[0])
                }
                tracer.end_span(span_id, status="success", usage=usage)
            except Exception as e:
                tracer.end_span(span_id, status="error", error=str(e))
                raise RuntimeError(f"LocalVLM transformers execution failed: {e}")

        # Store interaction in memory
        if session_id and memory:
            await memory.add_interaction(session_id, prompt, response)
            
        return response

    async def generate(self, prompt: str, session_id: Optional[str] = None) -> str:
        """Text-only generation fallback for VLMs."""
        if not self.hf_model and not self.is_ollama:
            if not config.LOAD_LOCAL_VLM:
                raise RuntimeError(
                    f"Attempted to use LocalVLM ({self.model}) but LOAD_LOCAL_VLM is False. "
                    "Initialize Phoenix AI with vlm=True to enable it."
                )
            await self.init()
            
        # Handle Memory
        memory = None
        try:
            memory = container.get("memory")
        except KeyError:
            pass

        context_prefix = ""
        messages = []
        if session_id and memory:
            semantic_context = await memory.search_memory(session_id, prompt)
            if semantic_context:
                context_prefix = f"Context from previous interactions:\n{semantic_context}\n\n"
            
            history = await memory.history.get(session_id)
            for item in history:
                messages.append(item["content"])

        messages.append({"role": "user", "content": f"{context_prefix}{prompt}"})

        span_id = tracer.start_span("LocalVLM.generate", {"model": self.model, "engine": "ollama" if self.is_ollama else "transformers"})

        response = ""
        if self.is_ollama:
            try:
                ollama_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"
                response = await self._ollama_generate(ollama_prompt, image_path=None)
                words = (len(ollama_prompt.split()) + len(response.split()))
                usage = {"total_tokens": int(words * 1.3)}
                tracer.end_span(span_id, status="success", usage=usage)
            except Exception as e:
                tracer.end_span(span_id, status="error", error=str(e))
                raise
        else:
            try:
                import torch
                if hasattr(self.processor, "apply_chat_template"):
                    text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = self.processor(text=[text], return_tensors="pt").to(self.hf_model.device)
                else:
                    plain_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
                    inputs = self.processor(text=plain_text, return_tensors="pt").to(self.hf_model.device)
                    
                with torch.no_grad():
                    generated_ids = self.hf_model.generate(**inputs, max_new_tokens=config.SECURITY_MAX_OUTPUT_LENGTH)
                    
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response = output[0]
                
                usage = {
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": len(generated_ids_trimmed[0]),
                    "total_tokens": inputs.input_ids.shape[1] + len(generated_ids_trimmed[0])
                }
                tracer.end_span(span_id, status="success", usage=usage)
            except Exception as e:
                tracer.end_span(span_id, status="error", error=str(e))
                raise RuntimeError(f"LocalVLM text generate failed: {e}")

        # Store interaction
        if session_id and memory:
            await memory.add_interaction(session_id, prompt, response)

        return response

    async def _ollama_generate(self, prompt: str, image_path: str = None) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if image_path:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                payload["images"] = [image_data]
                
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.base_url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"LocalVLM (Ollama) call failed: {e}")

