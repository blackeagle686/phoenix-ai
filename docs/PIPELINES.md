# Data Source Pipelines

The **RAGPipeline** is designed to ingest knowledge from a wide variety of sources. This guide explains how to use each pipeline effectively.

## 1. Local Files & Directories
Index local documents ranging from text files to complex codebases.

```python
# Ingest a single file
await rag.ingest("./docs/specification.pdf")

# Ingest an entire directory (recursive)
await rag.ingest("./src/core")
```
**Supported Extensions:**
- **Documents:** `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`, `.md`, `.txt`
- **Source Code:** `.py`, `.js`, `.ts`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.sql`, `.sh`

---

## 2. GitHub Repositories
Automatically clone and index entire remote repositories.

```python
# Clones and indexes the main branch
await rag.ingest_github("https://github.com/user/repo")

# Specify a branch
await rag.ingest_github("https://github.com/user/repo", branch="develop")
```
> [!NOTE]
> This requires `git` to be installed on the host system.

---

## 3. Web Scraping
Convert any public website into knowledge chunks.

```python
await rag.ingest_url("https://docs.Phoenix AI.ai/intro")
```

---

## 4. SQL Databases
Ingest structured data from relational databases.

```python
conn = "postgresql://user:pass@localhost/db"
query = "SELECT title, content FROM blog_posts"
await rag.ingest_sql(conn, query, text_column="content")
```

---

## 5. External JSON APIs
Connect to any REST API and index its responses.

```python
await rag.ingest_api(
    url="https://api.example.com/v1/news",
    method="GET",
    data_path="articles" # Path to extract from JSON
)
```

