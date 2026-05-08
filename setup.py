from setuptools import setup, find_packages

setup(
    name="phx-ashborn",
    version="0.2.5",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "requests>=2.28.0",
        "openai>=1.0.0",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.11.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pypdf>=3.10.0",
        "python-docx>=0.8.11",
        "qdrant-client>=1.6.0",
        "redis>=4.5.0",
        "celery>=5.3.0",
        "sqlalchemy>=2.0.0",
        "django>=5.0.0",
        "pandas",
        "openpyxl",
        "gTTS",
        "SpeechRecognition",
        "pydub",
        "Pillow",
        "psutil",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pyserial>=3.5",
        "paho-mqtt>=1.6.0"
    ],
    extras_require={
        "full": [
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes"
        ]
    },
    author="blackeagle686",
    description="Advanced AI Infrastructure SDK for Agentic Applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
