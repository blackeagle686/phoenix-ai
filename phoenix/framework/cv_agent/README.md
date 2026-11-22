# `phx-micron-cv` — Standalone Micron CV Agent

Ultra-lightweight, **zero-dependency** Computer Vision edge agent from Phoenix AI (`phoenix.framework.cv_agent`), packaged so you can install **only** the Micron CV Agent on **mobile phone CPUs (Android/Termux), Raspberry Pi, and very low-RAM devices** without installing all of `phx-ashborn` or any heavy packages.

## Highlights

- **Zero Core Dependencies**: Uses Python stdlib only (`sqlite3`, `dataclasses`, `collections.deque`, `json`, `urllib.request`) — `< 2 MB` RAM footprint and `~25 KB` package size.
- **Strict `<= 1B` MoE Only**: Enforces Mixture-of-Experts models `<= 1.0B` (e.g. `ibm-granite/granite-3.1-1b-a400m-instruct`, `micro-moe-500m-a150m`, `micro-moe-300m-a100m`, or custom `<= 1B` MoE GGUF files) or `model=None` zero-RAM mode.
- **Same Import Path**: `from phoenix.framework.cv_agent import CVAgent` — compatible seamlessly with the full `phx-ashborn` SDK.

## Installation (Without All `phx-ashborn`)

```bash
# 1. Standalone Micron CV Agent ONLY (0 dependencies, ~25 KB!)
pip install "git+https://github.com/blackeagle686/phoenix-ai.git#subdirectory=phoenix/framework/cv_agent"
# (Or from PyPI once published: `pip install phx-micron-cv`)

# 2. Standalone + quantized <=1B MoE GGUF CPU backend (for Phone / ARM CPU)
pip install "phx-micron-cv[gguf] @ git+https://github.com/blackeagle686/phoenix-ai.git#subdirectory=phoenix/framework/cv_agent"
```

If you already use `phx-ashborn`, you can also install the lightweight extra without heavy agent/RAG/ML dependencies:
```bash
pip install "phx-ashborn[cv]"       # or `pip install "phx-ashborn[cv-gguf]"`
```

Full documentation & real-world scenarios: [docs/micron_cv/](https://github.com/blackeagle686/phoenix-ai/tree/master/docs/micron_cv)
