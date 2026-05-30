# 🧠 phoenix (أستطيع قراءة عقلك)

تطوير بنية تحتية برمجية (SDK) جاهزة للإنتاج، مصممة لخدمات الذكاء الاصطناعي المبنية بلغة بايثون.

سواء كنت تبني باستخدام FastAPI أو Django أو أي خدمة مخصصة، فإن **phoenix** يزيل عنك عناء الإعدادات المتكررة. يوفر نظاماً موحداً وقابلاً للتبديل للتخزين المؤقت (Caching)، الوصول لقواعد البيانات، المهام الخلفية، دمج نماذج اللغة (LLM)، قواعد البيانات المتجهة (Vector DB)، وخطوط إنتاج الـ RAG.

## 🏗️ تدفق الهندسة المعمارية

تم بناء الـ SDK بالكامل حول فلسفة **"كل شيء كخدمة"** و **"الواجهة أولاً"**. يتم إدارة الخدمات مركزياً بواسطة نظام حقن التبعية (Dependency Injection)، مما يضمن نمطية كاملة وتجنب تضارب الحالات العالمية.

```mermaid
graph TD
    APP["المتصفح / التطبيق (FastAPI/Django)"] -->|"تهيئة"| INIT("Phoenix AI.py Initializer")
    APP -->|"طلب خدمة"| DI{"حاوية حقن التبعية"}
    INIT -->|"تسجيل الخدمات"| DI
    
    subgraph Core
        DI
        CONF["الإعدادات"]
        LFC["Lifecycle Hooks"]
    end
    
    subgraph البنية التحتية العامة
        DI -->|"يوفر"| CACHE["خدمة التخزين (Redis)"]
        DI -->|"يوفر"| DB["خدمة البيانات (SQLAlchemy)"]
        DI -->|"يوفر"| QUEUE["خدمة المهام (Celery)"]
    end

    subgraph مصادر البيانات
        DS_GITHUB["مستودعات GitHub"]
        DS_WEB["كشط الويب"]
        DS_SQL["قواعد بيانات SQL"]
        DS_API["واجهات البرمجة APIs"]
        DS_FILE["ملفات وأكواد محلية"]
    end
    
    subgraph عمليات الذكاء الاصطناعي
        DI -->|"يوفر"| LLM["خدمة النماذج اللغوية (OpenAI/Local)"]
        DI -->|"يوفر"| VLM["خدمة الرؤية (OpenAI/Local)"]
        DI -->|"يوفر"| AUDIO["خدمة الصوت (STT/TTS)"]
        DI -->|"يوفر"| VDB["قواعد البيانات المتجهة"]
        DI -->|"يوفر"| RAG["خط إنتاج RAG"]
        DI -->|"يوفر"| INSIGHT["محرك الاستبصار"]
        DI -->|"يوفر"| MEMORY["خدمة الذاكرة (سياق المحادثة)"]
        
        RAG -->|"استرجاع السياق من"| VDB
        RAG -->|"توليد الإجابة عبر"| LLM
        RAG -.->|"تحسين الاستعلام عبر"| MEMORY
        
        LLM -->|"مدعوم بـ"| MEMORY
        VLM -->|"مدعوم بـ"| MEMORY
        MEMORY -->|"تخزين/استرجاع الحقائق"| VDB
        
        INSIGHT -->|"استرجاع"| VDB
        INSIGHT -->|"توليد"| LLM
        INSIGHT -.->|"تخزين مؤقت"| CACHE
        
        DS_GITHUB --> RAG
        DS_WEB --> RAG
        DS_SQL --> RAG
        DS_API --> RAG
        DS_FILE --> RAG
    end

    subgraph المراقبة (Observability)
        OBS["المسجل والمتبع"] -.->|"يراقب"| CACHE
        OBS -.->|"يراقب"| DB
        OBS -.->|"يراقب"| LLM
        OBS -.->|"يراقب"| VLM
        OBS -.->|"يراقب"| AUDIO
        OBS -.->|"يراقب"| VDB
        OBS -.->|"يراقب"| MEMORY
    end
    
    subgraph الوكيل المستقل (Autonomous Agent)
        AGENT["Agent Core & Loop"]
        COGNITION["الإدراك (Thinker, Planner, Reflector)"]
        EXEC["التنفيذ (Actor, ToolManager)"]
        TOOLS_REG["الأدوات (WebSearch, CodeExecution)"]
        HYBRID_MEM["الذاكرة الهجينة"]
        
        AGENT --> COGNITION
        AGENT --> EXEC
        AGENT --> HYBRID_MEM
        EXEC --> TOOLS_REG
        AGENT -.->|"يستخدم"| LLM
    end
```

## 🚀 المتطلبات الرئيسية والميزات

1. **حقن التبعية (DI)**: سجل مركزي موحد للخدمات.
2. **الواجهة أولاً**: كل وحدة تلتزم بعقد أساسي غير متزامن.
3. **قواعد بيانات متجهة مرنة**: دعم أصلي لـ **ChromaDB** و **Qdrant**.
4. **تضمينات مدمجة**: مهيأة مسبقاً مع `sentence-transformers` لتوليد التضمينات محلياً.
5. **تنسيق RAG**: نظام `RAGPipeline` شامل يعالج تحميل المستندات (.pdf, .docx, .xlsx)، قواعد بيانات SQL، واجهات البرمجة الخارجية (APIs)، وكشط الويب.

## ⚠️ متطلبات الأجهزة للموديلات المحلية
إذا كنت تخطط لاستخدام التشغيل المحلي (Ollama أو Transformers)، يرجى التأكد من أن نظامك يلبي هذه المواصفات:
- **الذاكرة العشوائية (RAM)**: 8 جيجابايت كحد أدنى (يوصى بـ 16 جيجابايت+).
- **كارت الشاشة (GPU)**: ذاكرة 4 جيجابايت+ مطلوبة لموديلات الرؤية (VLM) باستخدام تقليب 4-bit.
- **المساحة**: 10 جيجابايت+ مساحة حرة لتخزين الموديلات.

> [!WARNING]
> النماذج ذات الموارد العالية قد تسبب عدم استقرار النظام على الأجهزة ذات الذاكرة المنخفضة أو التي تعتمد على المعالج فقط. يتبع الـ SDK نهج السلامة أولاً وسيطلب التأكيد قبل بدء تشغيل الموديلات المحلية.

## 📦 التثبيت

### 1. نسخ المستودع (Clone)
قم بنسخ المستودع وتثبيت التبعيات:

```bash
git clone https://github.com/blackeagle686/phoenix-ai.git
cd phoenix
pip install -r requirements.txt
```

### 2. تثبيت Pip محلي (اختياري)
إذا كنت ترغب في تثبيته كحزمة في بيئتك المحلية:
```bash
pip install .
```

## �️ البدائل الديناميكية وبايثون الأصلي (Native PyTorch)

يتضمن phoenix بنية تنسيق قوية مبنية على مبدأ "الإخفاق بوضوح والتعافي بلطف" (Fail-loud and recover gracefully) لمقدمي خدمات الذكاء الاصطناعي:

### 1. بدائل تفاعلية لمقدم الخدمة (Interactive Fallbacks)
عند إخفاق مزود الخدمة الأساسي (مثل Local) في الاتصال أو تعطله، تقوم طبقة التنسيق في SDK (`VLMPipeline` / `InsightEngine`) باعتراض الفشل فوراً وسؤالك عما إذا كنت تريد التحويل إلى المزود الاحتياطي (مثل OpenAI)، مما يتجاوز أعطال النظام بشكل كامل.

### 2. تخزين مؤقت أصلي لبايثون (`LocalVLM` و `LocalLLM`)
لا يوجد خادم `Ollama`؟ لا توجد مشكلة! يكتشف المزودون المحليون تلقائياً ما إذا كانت مكتبة `transformers` مثبتة، ويقومون بتشغيل النماذج محلياً مباشرةً على كارت الشاشة (GPU) باستخدام نمط Singleton محسّن.
> **نصيحة لمستخدمي Jupyter/Colab**: إذا واجهت تحذيرات `Ollama` مستمرة بعد تثبيت `transformers`، قم بتشغيل `LocalVLM._model_cache.clear()` أو `LocalLLM._model_cache.clear()` في دفتر الملاحظات الخاص بك لمسح الحالة السابقة وإجبار التحميل باستخدام بيئة PyTorch.

### 3. التكميم التلقائي 4-Bit (Quantization)
لمنع أخطاء نفاد ذاكرة كارت الشاشة (`CUDA Out of Memory`) على الكروت الصغيرة (مثل كروت T4 في Colab)، يكتشف الـ SDK تلقائياً وجود مكتبة `bitsandbytes` (قم بتثبيتها `pip install bitsandbytes`) ويطبق فوراً خاصية `load_in_4bit=True` لتقليص النماذج الضخمة (مثل Qwen2-VL) لتناسب مساحة ذاكرتك المؤقتة.

### 4. استيعاب مرن لملفات الـ PDF
طريقة `RAGPipeline.ingest()` تدعم قراءة ملفات PDF بطريقة موثوقة للغاية من خلال الفحص التسلسلي لمكتبات القراءة المتوفرة: `pypdf`، `pymupdf` (باستيراد `fitz`)، `pdfplumber`، و `PyPDF2`. ببساطة، قم بتثبيت المكتبة التي تفضلها (نوصي بـ `pip install pymupdf` من أجل السرعة) وسيعمل النظام بسلاسة تامة!

## 🚀 وضع الإطار (Framework Mode): روبوت دردشة عالي المستوى

يتضمن Phoenix AI SDK الآن **طبقة إطار عمل (Framework Layer)** عالية المستوى تسمح لك ببناء عملاء ذكاء اصطناعي معقدين مع رؤية، وRAG، وذاكرة في **سطر واحد فقط من الكود**.

```python
from phoenix import ChatBot

# بناء عميل متكامل مع الحماية وإعدادات مخصصة
bot = (ChatBot(local=True, vlm=True)
       .with_rag(["./docs", "./src"])       # مجلدات أو ملفات
       .with_memory()                       # تفعيل ذاكرة الجلسة
       .with_security(mode="strict")        # الحماية من حقن الأوامر (Prompt Injection)
       .with_system_prompt("Expert Dev")    # توجيه سلوك البوت
       .build())

# أو الانتقال إلى OpenAI بسطر واحد
# bot.with_openai(api_key="sk-...", base_url="https://api.openai.com")

# تفاعل متعدد الوسائط
response = await bot.chat("ماذا يوجد في هذه الصورة؟", image_path="vision.jpg")
print(response) 
```
> [!TIP]
> استخدم `.set_session("user_123")` على مثيل البوت للتبديل بين المستخدمين المختلفين في بيئات الإنتاج مثل FastAPI.

## 🤖 وضع الإطار (Framework Mode): الوكيل المستقل (Autonomous Agent)

يدعم Phoenix AI SDK الآن إنشاء وكيل مستقل بالكامل يمكنه التفكير، والتخطيط، وتنفيذ الأدوات، والتأمل في تقدمه باستخدام سطر واحد فقط من الكود! 

> [!TIP]
> للحصول على تعمق في الهندسة المعمارية وأنماط التكامل (CLI, Web, GUI)، راجع **[دليل إطار عمل الوكيل](./docs/AGENT_GUIDE.md)** (متوفر باللغة الإنجليزية حالياً).

#### ⚡ محرك إدراكي عالي السرعة

```python
import asyncio
from phoenix import Agent
from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.memory.hybrid import HybridMemory
from phoenix.framework.agent.tools.registry import ToolRegistry

async def agent_demo():
    # تهيئة الوكيل باستخدام المكونات الافتراضية
    agent = Agent(
        llm=OpenAILLM(), 
        memory=HybridMemory(), 
        tools=ToolRegistry.load_default()
    )
    
    # تشغيل مهمة مستقلة
    result = await agent.run("قم بتحليل آخر الأخبار حول الذكاء الاصطناعي.")
    print(f"المخرجات النهائية للوكيل: {result}")
```

### 🧰 أدوات مخصصة وإدخال/إخراج الملفات

يأتي الوكيل مزوداً مسبقاً بمجموعة قوية من الأدوات الافتراضية، بما في ذلك `python_repl`، `web_search`، `file_read`، `file_write`، و `file_search`.

يمكنك أيضاً بسهولة إنشاء وحقن أدواتك المخصصة باستخدام المزخرف `@tool`:

```python
from phoenix.framework.agent import tool

# 1. تعريف منطق الأداة المخصصة
@tool(name="custom_math", description="Calculates the square of a given number. Input: 'number' (int).")
def custom_math_tool(number: int):
    return f"The square of {number} is {number ** 2}"

# 2. تسجيلها مباشرة إلى الوكيل
agent.register_tool(custom_math_tool)

# 3. يمكن للوكيل الآن استخدام 'custom_math' بشكل مستقل في تخطيطه!
await agent.run("ما هو مربع العدد 12؟")
```

### ⚡ أوضاع التنفيذ (التوجيه التلقائي)

يتميز الوكيل بتوجيه ذكي لتوفير الوقت وتكاليف الـ API في المهام البسيطة. يعمل الوكيل افتراضياً في وضع `mode="auto"`.
- **`auto`**: يحلل الوكيل الموجه. إذا كان سؤالاً بسيطاً، يقدم إجابة مباشرة. وإذا كان يتطلب أدوات أو منطقاً متعدد الخطوات، يقوم بتشغيل حلقة التخطيط.
- **`fast_ans`**: يجبر الوكيل على تخطي التخطيط والإجابة فوراً باستخدام سياق الذاكرة.
- **`plan`**: يجبر الوكيل على الدخول في حلقة "التفكير -> التخطيط -> التنفيذ -> التأمل" الصارمة.

```python
# يجبر الوكيل على إجابة سريعة (يتخطى تنفيذ الأدوات)
await agent.run("مرحباً، من أنت؟", mode="fast_ans")

# يجبر الوكيل على التخطيط المعقد
await agent.run("ابحث في الويب عن أحدث إصدارات لغة بايثون...", mode="plan")
```

## 📖 التشغيل السريع: RAG Pipeline

نظام `RAGPipeline` هو أعلى مستوى خدمة للتعامل مع المعرفة القائمة على المستندات.

```python
import asyncio
from phoenix import init_phoenix, startup_phoenix, get_rag_pipeline

async def rag_demo():
    init_phoenix()
    await startup_phoenix()
    rag = get_rag_pipeline()

    # 1. استيعاب المستندات (يدعم المستندات + الأكواد المصدرية .py, .js, .go, .rs, إلخ)
    await rag.ingest("./my_project")

    # 2. استيعاب من مستودع GitHub (نسخ وفهرسة تلقائية)
    await rag.ingest_github("https://github.com/blackeagle686/phoenix-ai.git")

    # 3. استيعاب من رابط ويب
    await rag.ingest_url("https://example.com/docs/api")

    # 4. الاستعلام مع مراجع تلقائية (Citations)
    answer = await rag.query("كيف يمكنني توسيع طبقة التخزين المؤقت؟")
    print(f"إجابة الذكاء الاصطناعي: {answer}")
```

## 🖼️ التشغيل السريع: الرؤية (VLM)

يقوم `VLMPipeline` بتنسيق مهام الرؤية مع التخزين المؤقت التلقائي و RAG.

```python
from phoenix import init_phoenix_full, get_vlm_pipeline

async def vision_demo():
    await init_phoenix_full()
    vlm = get_vlm_pipeline()

    # متكامل: تخزين النتائج + حقن سياق RAG
    answer = await vlm.ask("ماذا يوجد في هذه الصورة؟", "image.png", use_rag=True)
    print(answer)
```

