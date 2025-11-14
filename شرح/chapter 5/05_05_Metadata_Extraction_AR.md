# 🗃️ البيانات الوصفية للعقد (Metadata for Nodes)

<div dir="rtl">

البيانات الوصفية بتوفر سياق أو معلومات إضافية عن العقد.

خلال الاسترجاع نقدر نستفيد من السياق والمعلومات الإضافية دي، لاسترجاع أدق وأكثر صلة. بس، فعالية الطريقة دي بتعتمد على جودة وملاءمة علامات البيانات الوصفية المستخدمة. أبسط طريقة لإضافة البيانات الوصفية هي إنك تعملها يدوياً.

خلينا نضيف شوية بيانات وصفية عن اللي كل واحد من السينباي بتوعنا معروف بيه.

```python
known_for = {
    "Naval Ravikant": "معروف برؤاه حول إزاي تبني الثروة وتحقق السعادة من خلال تطوير المعرفة الخاصة، تبني المسؤولية، لعب ألعاب طويلة المدى، وفهم قوة الفائدة المركبة في كل مجالات الحياة.",
    "Balaji Srinivasan": "عنده رؤى حول إزاي تفكر بشكل مستقل، تحدد الفرص، وتبني مستقبل أفضل من خلال التطبيق الاستراتيجي للتكنولوجيا والتفكير الواضح.",
    "Paul Graham": "بيقدم نصايح عن عقلية الهاكر، بيجادل إن الهاكرز حقيقةً صانعين ومبدعين - زي الرسامين - واللي يقدروا يستفيدوا من طريقة تفكيرهم الفريدة لدفع الحدود، تحدي الوضع الراهن، وتشكيل المستقبل من خلال التكنولوجيا وريادة الأعمال.",
    "Nassim Nicholas Taleb": "بيجادل لـ 'المخاطرة في اللعبة'، يعني إن يكون عندك حصة شخصية في النتيجة ضروري للعدالة لأنه بيوازن الحوافز وبيعرض الأفراد لكل من المكافآت والمخاطر المحتملة لقراراتهم.",
    "Seneca": "بيقدم نصايح خالدة حول إزاي تزرع الحكمة، تبني المرونة الذهنية، وتعيش حياة هادفة وراضية من خلال التركيز على الأساسي، إتقان المشاعر، ومواءمة نفسك مع الطبيعة.",
    "Bruce Lee": "بيقدم حكمة عميقة حول تحسين الذات، النمو الشخصي، وفلسفة الفنون القتالية، مع التأكيد على أهمية القدرة على التكيف، التعبير عن الذات، وتبني طريقك الفريد في الحياة."
}
```

```python
for document in senpai_documents:
    document.metadata['known_for'] = known_for.get(document.metadata['author'])
```

```python
senpai_documents[42].metadata
```

## استخراج البيانات الوصفية تلقائياً

استخراج البيانات الوصفية في LlamaIndex هي عملية بتساعد في توضيح الفرق بين الفقرات النصية اللي بتبان متشابهة، خصوصاً في المستندات الطويلة.

ده بيتحقق باستخدام نماذج اللغة الكبيرة (LLMs) لاستخراج معلومات سياقية ذات صلة بالمستند. المعلومات دي بتساعد نماذج الاسترجاع واللغة في التمييز بين الفقرات المتشابهة.

في LlamaIndex، استخراج البيانات الوصفية بيتم باستخدام مستخرجات ميزات مختلفة ضمن كلاس [`MetadataExtractor`](https://github.com/run-llama/llama_index/tree/954398e1957027a364d0d332fee61733ad322f8b/llama-index-core/llama_index/core/extractors).

المستخرجات دي بتتضمن:

- `SummaryExtractor`: المستخرج ده بيولد تلقائياً ملخص على مجموعة من العقد.

- `QuestionsAnsweredExtractor`: المستخرج ده بيحدد مجموعة من الأسئلة اللي كل عقدة تقدر تجاوب عليها.

- `TitleExtractor`: المستخرج ده بيحدد عنوان على سياق كل عقدة.

- `KeywordExtractor`: الكلمات المفتاحية اللي بتحدد العقدة بشكل فريد.

```python
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor
```

```python
print(SummaryExtractor().prompt_template)
```

```python
print(QuestionsAnsweredExtractor().prompt_template)
```

```python
print(TitleExtractor().node_template)
```

### KeywordExtractor عنده قالب الـ prompt مدفون في استدعاء LLM، ومش سمة.

ده اللي هو في [الكود المصدري](https://github.com/run-llama/llama_index/blob/954398e1957027a364d0d332fee61733ad322f8b/llama-index-core/llama_index/core/extractors/metadata_extractors.py#L198):

```python
f"""\
{{context_str}}. أعطي {self.keywords} كلمات مفتاحية فريدة للمستند ده. 
التنسيق: مفصولة بفواصل. الكلمات المفتاحية:
```

## استخراج البيانات الوصفية التلقائي

خلينا نعمل شوية استخراج تلقائي للبيانات الوصفية لنتائج استرجاع أحسن.

هنستخدم مستخرجين:

- `QuestionAnsweredExtractor` لتوليد أزواج سؤال/إجابة من قطعة نص

- `SummaryExtractor` لاستخراج الملخصات، مش بس ضمن النص الحالي، لكن كمان ضمن النصوص المجاورة.

الاستراتيجية دي بتؤدي لإجابة جودة أعلى بناءً على النتائج المستَرجعة.

لعمل ده، بنعرف مستخرجات البيانات الوصفية:

- `qa_extractor`

- `summary_extractor`

لاحظ استخدام `MetadataMode.EMBED` ده بيحدد إزاي البيانات الوصفية بتتعامل لما تولد تضمينات لمستند أو عقدة. لما تستدعي دالة `get_content()` على مستند وتحدد `MetadataMode.EMBED`، بترجع محتوى المستند مع البيانات الوصفية المرئية لنموذج التضمين.

كمان هنستخدم `GPT-3.5-Turbo` لتوليد البيانات الوصفية.

#### 👨🏽‍💻 بشجعك تجرب مستخرجات البيانات الوصفية التانية وتشوف النتائج بتاعتك شكلها إيه.

مثلاً، تقدر تجرب `KeywordExtractor` أو `TitleExtractor` كده:

```python
keyword_extractor = KeywordExtractor(keywords=10, llm=llm)

title_extractor = TitleExtractor(nodes=5, llm=llm)
```

```python
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.llms.openai import OpenAI

qa_llm = OpenAI(model="gpt-4o")

text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=16)

qa_extractor = QuestionsAnsweredExtractor(
    questions=2, 
    llm=qa_llm, 
    metadata_mode=MetadataMode.EMBED,
    embed_model=Settings.embed_model,
)

summary_extractor = SummaryExtractor(
    summaries=["prev", "self", "next"], 
    llm=qa_llm,
)
```

### 👷🏽‍♂️ 🗂️ الإدخال لـ Qdrant وبناء الفهرس

في الفيديوهات القليلة اللي فاتت عملنا تقسيم العقد الأول وبعدين أدخلنا لـ Qdrant. ده كان عشان نوضح ليك النمط ونديك إحساس بإزاي التقسيم بيشتغل.

بس، نقدر فعلياً نعمل النوع ده من الحاجات مباشرة باستخدام خط أنابيب الإدخال.

ملحوظة، هسيب ليك إنك تجرب باستخدام واحد، أو كلا المستخرجين وتعبث بالمعاملات الفائقة.

التحليل هنا أخد حوالي 30 دقيقة.

```python
from llama_index.core import StorageContext
from llama_index.core.settings import Settings

from utils import create_index, create_query_engine, ingest, setup_vector_store

COLLECTION_NAME = "words-of-the-senpai-qa-plus-summaries-nodes"

qa_summaries_vector_store = setup_vector_store(QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME)

transforms = [text_splitter, qa_extractor, summary_extractor, Settings.embed_model]

qa_summaries = ingest(
    documents=senpai_documents,
    transformations=transforms,
    vector_store=qa_summaries_vector_store
)

qa_summaries_index = create_index(
    from_where="vector_store",
    vector_store=qa_summaries_vector_store,
    embed_model=Settings.embed_model,
)
```

```python
len(qa_summaries)
```

```python
qa_summaries[100].__dict__
```

```python
print(qa_summaries[100].get_content(metadata_mode="all"))
```

### 🔧 إعداد محرك الاستعلام وخط الأنابيب

```python
from llama_index.core import PromptTemplate
from utils import create_query_engine
from prompts import HYPE_ANSWER_GEN_PROMPT

HYPE_ANSWER_GEN_PROMPT_TEMPLATE = PromptTemplate(HYPE_ANSWER_GEN_PROMPT)

qa_summaries_query_engine = create_query_engine(
    index=qa_summaries_index, 
    mode="query",
    response_mode="compact",
    similiarty_top_k=5,
    vector_store_query_mode="mmr", 
    vector_store_kwargs={"mmr_threshold": 0.42},
)

qa_summaries_query_engine.update_prompts({'response_synthesizer:text_qa_template': HYPE_ANSWER_GEN_PROMPT_TEMPLATE})
```

```python
from utils import create_query_pipeline
from llama_index.core.query_pipeline import InputComponent

input_component = InputComponent()

qa_summaries_chain = [input_component, qa_summaries_query_engine]

qa_summaries_query_pipeline = create_query_pipeline(qa_summaries_chain)
```

```python
qa_summaries_query_pipeline.run(input="إزاي أقدر أضمن اتخاذ قرارات حاسمة في حياتي؟")
```

</div>
