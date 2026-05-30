# خطوط استيعاب البيانات (Data Source Pipelines)

تم تصميم **RAGPipeline** لاستيعاب المعرفة من مجموعة واسعة من المصادر. يوضح هذا الدليل كيفية استخدام كل خط أنابيب بشكل فعال.

## 1. الملفات والمجلدات المحلية
فهرسة المستندات المحلية بدءاً من الملفات النصية إلى الأكواد المصدرية المعقدة.

```python
# استيعاب ملف واحد
await rag.ingest("./docs/specification.pdf")

# استيعاب مجلد كامل (بشكل متكرر)
await rag.ingest("./src/core")
```
**الامتدادات المدعومة:**
- **المستندات:** `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`, `.md`, `.txt`
- **الأكواد المصدرية:** `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.c`, `.cpp`, `.h`, `.sql`, `.sh`

---

## 2. مستودعات GitHub
نسخ وفهرسة مستودعات كاملة عن بعد تلقائياً.

```python
# نسخ وفهرسة الفرع الرئيسي (main)
await rag.ingest_github("https://github.com/user/repo")

# تحديد فرع معين
await rag.ingest_github("https://github.com/user/repo", branch="develop")
```
> [!NOTE]
> يتطلب هذا تثبيت أداة `git` على النظام المضيف.

---

## 3. استخراج بيانات الويب (Web Scraping)
تحويل أي موقع ويب عام إلى قطع معرفية.

```python
await rag.ingest_url("https://docs.Phoenix AI.ai/intro")
```

---

## 4. قواعد بيانات SQL
استيعاب البيانات المنظمة من قواعد البيانات العلاقاتية.

```python
conn = "postgresql://user:pass@localhost/db"
query = "SELECT title, content FROM blog_posts"
await rag.ingest_sql(conn, query, text_column="content")
```

---

## 5. واجهات JSON API الخارجية
الاتصال بأي واجهة برمجة تطبيقات REST وفهرسة استجاباتها.

```python
await rag.ingest_api(
    url="https://api.example.com/v1/news",
    method="GET",
    data_path="articles" # المسار لاستخراج البيانات من JSON
)
```

