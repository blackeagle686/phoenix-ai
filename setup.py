from setuptools import setup, find_packages

EXTRAS = {
    "agent": [
        "pydantic",
        "redis>=4.5.0",
        "celery>=5.3.0",
        "duckduckgo-search",
        "chromadb>=0.4.0",
        "qdrant-client>=1.6.0"
    ],
    "chatbot": [
        "markdown>=3.4.0",
        "beautifulsoup4>=4.11.0",
        "pypdf>=3.10.0",
        "python-docx>=0.8.11",
        "sqlalchemy>=2.0.0",
        "django>=5.0.0",
        "pandas",
        "openpyxl",
        "Pillow",
        "fastapi",
        "uvicorn",
        "python-multipart"
    ],
    "sensorium": [
        "pyserial>=3.5",
        "paho-mqtt>=1.6.0",
        "psutil"
    ],
    "local": [
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers>=2.2.0"
    ],
    "audio": [
        "gTTS",
        "SpeechRecognition",
        "pydub"
    ],
    "vector": [
        "chromadb>=0.4.0",
        "qdrant-client>=1.6.0"
    ],
    "rag": [
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pypdf>=3.10.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.0",
        "sqlalchemy>=2.0.0",
        "httpx>=0.24.0",
        "google-api-python-client>=2.0.0",
        "google-auth-oauthlib>=1.0.0",
        "redis>=4.5.0"
    ],
    "brain": [
        "maturin>=1.0.0"
    ],
    "ml": [
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "joblib>=1.3.0",
        "lightgbm>=4.0.0",
        "onnxruntime>=1.16.0"
    ],
    "cv": [],
    "cv-gguf": [
        "llama-cpp-python>=0.2.85"
    ]
}

EXTRAS["full"] = list(set(dep for k, deps in EXTRAS.items() if k != "cv-gguf" for dep in deps))

setup(
    name="phx-ashborn",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "openai>=1.0.0",
        "pydantic",
        "httpx>=0.24.0",
        "numpy"
    ],
    extras_require=EXTRAS,
    author="blackeagle686",
    description="Advanced AI Infrastructure SDK for Agentic Applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
