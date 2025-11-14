# فهرس ملخص المستند (Document Summary Index)

<div dir="rtl">

<img src="https://docs.llamaindex.ai/en/stable/_static/production_rag/decouple_chunks.png" style="width:50%; height:50%">

المصدر: [مستندات LlamaIndex](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/#decoupling-chunks-used-for-retrieval-vs-chunks-used-for-synthesis)

الطريقة دي بتستخرج ملخصات لكل مستند لتحسين أداء الاسترجاع على البحث الدلالي التقليدي على قطع النص بس. بتستخدم الملخصات الموجزة وقدرات الاستدلال لنماذج اللغة الكبيرة لتحسين الاسترجاع قبل التركيب على القطع المستَرجعة.

## 🚫 محدوديات الاسترجاع المبني على القطع

- القطع بتفتقر للسياق العام
- محتاجة ضبط دقيق لعتبات التشابه
- التضمينات ممكن متلتقطش الملاءمة كويس
- تصفية الكلمات المفتاحية عندها تحدياتها الخاصة

### 📝 فهرس ملخص المستند بيخزن

- ملخص مستخرَج بواسطة LLM لكل مستند
- المستند مقسم لقطع نص
- ربط بين الملخصات والمستندات/القطع المصدر

### 🔍 مناهج الاسترجاع

1. 🤖 مبني على LLM: اللغة الكبيرة بتعطي نقاط لملاءمة ملخصات المستندات

2. 📐 مبني على التضمين: استرجاع بناءً على تشابه تضمين الملخص

## ⚖️ المزايا

- الملخصات بتوفر سياق أكتر من القطع لوحدها
- اللغة الكبيرة تقدر تستنتج على الملخصات قبل المستندات الكاملة
- تمثيلات مثلى مختلفة للاسترجاع مقابل التركيب

## 🚀 التقنيات الأساسية

1. تضمين الملخصات المرتبطة بقطع المستند
2. استرجاع الملخصات، استبدالها بمحتوى المستند الكامل

## إعداد مخزن المتجهات

```python
from llama_index.core import StorageContext
from llama_index.core.settings import Settings

from utils import create_index, create_query_engine, ingest, setup_vector_store

COLLECTION_NAME = "words-of-the-senpai-document-summary-index"

doc_summary_vector_store = setup_vector_store(QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME)
```

## الإدخال باستخدام [`DocumentSummaryIndex`](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/document_summary/base.py)

الـ `DocumentSummaryIndex`:

- 📝 بيبني فهرس من مجموعة مستندات
- 🎯 بيولد ملخص لكل مستند باستخدام مُركب استجابة
- 💾 بيخزن الملخصات وعقد المستند المقابلة في الفهرس

### 🌐 الاسترجاع

- بيدعم وضعين للاسترجاع: مبني على التضمين ومبني على LLM
- 🪢 الاسترجاع المبني على التضمين:
  - بيضمن الملخصات باستخدام نموذج تضمين
  - بيسترجع الملخصات ذات الصلة بناءً على التشابه لتضمين الاستعلام

- 🧠 الاسترجاع المبني على LLM:
  - بيستخدم LLM لاسترجاع الملخصات ذات الصلة بناءً على استعلام

بيركز على فهرسة المستندات، توليد الملخصات، وتوفير طرق استرجاع فعالة بناءً إما على التضمينات أو LLMs. المستَرجع كمان بيدعم عمليات إدارة المستندات زي إضافة وحذف المستندات من الفهرس.

### الـ API عالي المستوى بيستخدم الاسترجاع المبني على التضمين افتراضياً.

```python
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer
from llama_index.core.node_parser import TokenTextSplitter

splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=16)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)

doc_summary_index = DocumentSummaryIndex.from_documents(
    senpai_documents,
    llm=Settings.llm,
    embed_model=Settings.embed_model,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
    vector_store=doc_summary_vector_store
)
```

### 🔧 إعداد محرك الاستعلام وخط الأنابيب

```python
from llama_index.core import PromptTemplate
from utils import create_query_engine
from prompts import HYPE_ANSWER_GEN_PROMPT

HYPE_ANSWER_GEN_PROMPT_TEMPLATE = PromptTemplate(HYPE_ANSWER_GEN_PROMPT)

doc_summaries_query_engine = create_query_engine(
    index=doc_summary_index, 
    mode="query",
    response_mode="compact",
    similiarty_top_k=5,
    vector_store_query_mode="mmr", 
    vector_store_kwargs={"mmr_threshold": 0.42},
)

doc_summaries_query_engine.update_prompts({'response_synthesizer:text_qa_template': HYPE_ANSWER_GEN_PROMPT_TEMPLATE})
```

ملحوظة: مش هنشغل الاستنتاج باستخدام اللي فوق لأني عايز أوريك الـ API منخفض المستوى للاسترجاع المبني على التضمين كمان. هنستخدم ده للتوليد.

## 📜 [مستَرجعات ملخص المستند](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/document_summary/retrievers.py)

<img src="https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2F6d78d199badf9b45f5637d2a87aee0b12b9a335c-2099x1134.png%3Ffit%3Dmax%26auto%3Dformat&w=1920&q=75" style="width:70%; height:70%">

المصدر: [مدونة LlamaIndex](https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec)

- 📂 بيحتوي على نوعين من المستَرجعات:
  1. 🧠 مستَرجع مبني على LLM (`DocumentSummaryIndexLLMRetriever`)
  2. 🎨 مستَرجع مبني على التضمين (`DocumentSummaryIndexEmbeddingRetriever`)

مستَرجعات ملخص المستند دي بتسترجع الملخصات ذات الصلة بكفاءة من فهرس ملخص مستند.

المستَرجع المبني على LLM بيستخدم نماذج اللغة لاختيار الملخصات ذات الصلة بناءً على استعلام، بينما المستَرجع المبني على التضمين بيستخدم تشابه التضمين لإيجاد الملخصات ذات الصلة.

### 🧠 [`DocumentSummaryIndexLLMRetriever`](https://github.com/run-llama/llama_index/blob/99984eb87afb2e7feda65d5246ad166b0042f6fe/llama-index-core/llama_index/core/indices/document_summary/retrievers.py#L28)

- 📜 بيسترجع الملخصات ذات الصلة من الفهرس باستخدام استدعاءات LLM
- 🎛️ prompt قابل للتخصيص لاختيار الملخصات ذات الصلة
- 🍰 بيعالج عقد الملخص على دفعات
- 🔝 بيسترجع أعلى k عقدة ملخص بناءً على تقييم ملاءمة LLM
- 🤖 بيستخدم LLM لاختيار الملخصات ذات الصلة

#### معاملات لازم تعرفها:

- `index`: الفهرس للاسترجاع منه.

- `choice_select_prompt`: الـ prompt المستخدم لاختيار الملخصات ذات الصلة. الـ prompt الافتراضي ممكن تلاقيه [هنا](https://github.com/run-llama/llama_index/blob/99984eb87afb2e7feda65d5246ad166b0042f6fe/llama-index-core/llama_index/core/prompts/default_prompts.py#L392)

- `choice_batch_size`: عدد عقد الملخص المراد إرسالها للـ LLM في وقت واحد. القيمة الافتراضية 10

- `choice_top_k`: عدد عقد الملخص المراد استرجاعها. القيمة الافتراضية 1.

- `format_node_batch_fn`: دالة لتنسيق دفعة من العقد لـ LLM. ده افتراضياً `default_format_node_batch_fn`، اللي بتنسق دفعة من عقد الملخص بتعيين رقم لكل عقدة وضم محتواها بفاصل.

- `parse_choice_select_answer_fn`: دالة لتحليل استجابة LLM. افتراضياً `default_parse_choice_select_answer_fn`، اللي بتحلل سلسلة الإجابة من LLM، باستخراج أرقام الإجابات المختارة ونقاط الملاءمة المقابلة، وبترجعهم كقوائم.

- `llm` (LLM): الـ llm المستخدم.

```python
from llama_index.core.indices.document_summary import DocumentSummaryIndexLLMRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
```

```python
doc_llm_retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    choice_top_k=5,
    llm=Settings.llm,
    # choice_select_prompt=None,
    # choice_batch_size=10,
    # format_node_batch_fn=None,
    # parse_choice_select_answer_fn=None,
)

doc_llm_query_engine = RetrieverQueryEngine(
    retriever=doc_llm_retriever,
    response_synthesizer=response_synthesizer,
)

doc_llm_query_engine.update_prompts({'response_synthesizer:text_qa_template': HYPE_ANSWER_GEN_PROMPT_TEMPLATE})
```

```python
doc_llm_query_engine.query("إزاي أقدر أوقف التحليل الزائد لمزاجي ومشاعري؟")
```

```python
from utils import create_query_pipeline
from llama_index.core.query_pipeline import InputComponent

input_component = InputComponent()

doc_llm__chain = [input_component, doc_llm_query_engine]

doc_llm_query_pipeline = create_query_pipeline(doc_llm__chain)
```

```python
doc_llm_query_pipeline.run(input="إزاي أقدر أوقف التحليل الزائد لمزاجي ومشاعري؟")
```

### 🎨 [`DocumentSummaryIndexEmbeddingRetriever`](https://github.com/run-llama/llama_index/blob/aad4a6fb94c8fcaf1b7dfac56b88b9e277886bfe/llama-index-core/llama_index/core/indices/document_summary/retrievers.py#L121)

- 📜 بيسترجع الملخصات ذات الصلة من الفهرس باستخدام تشابه التضمين
- 🔢 بيسترجع أعلى k عقدة ملخص بناءً على تشابه التضمين
- 🪢 بيستخدم نموذج تضمين لتضمين الاستعلام
- 📏 بيستعلم مخزن المتجهات لإيجاد ملخصات متشابهة

#### معاملات لازم تعرفها

- `index`: الفهرس للاسترجاع منه.

- `similarity_top_k`: عدد عقد الملخص المراد استرجاعها.

```python
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever

doc_embed_retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    # similarity_top_k=1,
)

doc_embed_query_engine = RetrieverQueryEngine(
    retriever=doc_embed_retriever,
    response_synthesizer=response_synthesizer,
)
```

```python
doc_embed__chain = [input_component, doc_embed_query_engine]

doc_embed_query_pipeline = create_query_pipeline(doc_embed__chain)
```

```python
doc_embed_query_pipeline.run(input="إزاي أقدر أوقف التحليل الزائد لمزاجي ومشاعري؟")
```

</div>
