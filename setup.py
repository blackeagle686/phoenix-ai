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
    ]
}

EXTRAS["full"] = list(set(dep for deps in EXTRAS.values() for dep in deps))

setup(
    name="phx-ashborn",
    version="0.2.9",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "openai>=1.0.0",
        "pydantic"
    ],
    extras_require=EXTRAS,
    author="blackeagle686",
    description="Advanced AI Infrastructure SDK for Agentic Applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
