# دليل خدمة ضبط النماذج (Fine-Tuning Service)

تسمح لك **خدمة ضبط النماذج** بتكييف النماذج اللغوية (LLMs) ونماذج الرؤية (VLMs) مع مجالك الخاص باستخدام العتاد المحلي أو واجهات برمجة التطبيقات السحابية.

## 1. الضبط المحلي (LoRA / QLoRA)
تدريب النماذج بكفاءة على وحدة معالجة الرسومات (GPU) الخاصة بك باستخدام تقنية Parameter-Efficient Fine-Tuning (PEFT).

### المتطلبات الأساسية
- `pip install peft trl bitsandbytes accelerate`
- وحدة معالجة رسومات (GPU) بذاكرة >= 8GB (للنماذج الصغيرة) أو >= 16GB (للنماذج بحجم 7B).

### الاستخدام
```python
from phoenix import init_phoenix_full, get_finetuner

async def train_locally():
    await init_phoenix_full()
    finetuner = get_finetuner(provider="local")

    # بدء التدريب باستخدام مجموعة بيانات JSONL
    # تنسيق البيانات: {"text": "Instruction: ... Context: ... Response: ..."}
    job_id = await finetuner.train(
        dataset_path="./data/my_dataset.jsonl",
        model_id="Qwen/Qwen2-1.5B-Instruct",
        output_dir="./training_outputs"
    )
    
    print(f"بدأ التدريب: {job_id}")
```

---

## 2. الضبط السحابي (OpenAI)
تكييف النماذج في السحاب دون الحاجة إلى موارد GPU محلية.

### الاستخدام
```python
from phoenix import init_phoenix_full, get_finetuner

async def train_cloud():
    await init_phoenix_full()
    finetuner = get_finetuner(provider="openai")

    # بدء مهمة سحابية
    job_id = await finetuner.train(
        dataset_path="./data/cloud_dataset.jsonl",
        model_id="gpt-4o-mini-2024-07-18"
    )
    
    # مراقبة الحالة
    status = await finetuner.get_status(job_id)
    print(f"حالة المهمة: {status['status']}")
```

---

## 3. تنسيق مجموعة البيانات
- **LLM (نص):** ملفات JSONL حيث كل سطر هو قاموس يحتوي على مفتاح `"text"` يضم المطالبة الكاملة والاستجابة.
- **VLM (رؤية):** دعم مجموعات البيانات (صورة-نص) يتبع تنسيقات Hugging Face القياسية.

## 4. الإعدادات
يمكنك تخصيص ما يلي في ملف `.env` الخاص بك:
```env
TRAINING_BATCH_SIZE=1
TRAINING_EPOCHS=3
TRAINING_LEARNING_RATE=2e-4
TRAINING_LORA_R=8
```

