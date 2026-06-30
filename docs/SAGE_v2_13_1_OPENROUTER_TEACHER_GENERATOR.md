# SAGE v2.13.1 - OpenRouter Teacher Dataset Generator

## Purpose

v2.13.1 uses OpenRouter API instead of the direct OpenAI Responses API.

OpenRouter endpoint:

```text
https://openrouter.ai/api/v1/chat/completions
```

This script uses the OpenAI Python SDK pointed at OpenRouter:

```python
AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

## Why OpenRouter

```text
One key
Many teacher model choices
Same SAGE JSON schema
Can switch model by changing SAGE_TEACHER_MODEL
```

## Setup

```powershell
pip install -r requirements-teacher.txt
$env:OPENROUTER_API_KEY="sk-or-..."
$env:SAGE_TEACHER_MODEL="openai/gpt-4o-mini"
$env:OPENROUTER_APP_TITLE="SAGE Teacher Dataset Generator"
```

## Generate seeds

```powershell
python tools\generate_seed_questions.py --limit 80
```

## Dry run

```powershell
python tools\generate_teacher_dataset_openrouter.py --seeds data\seeds\generated_seed_questions.jsonl --limit 3 --dry-run
```

## Generate examples

```powershell
python tools\generate_teacher_dataset_openrouter.py --seeds data\seeds\generated_seed_questions.jsonl --limit 50 --concurrency 3 --temperature 0.3
```

## Validate

```powershell
python tools\validate_teacher_dataset.py --file data\teacher\sage_teacher_examples.jsonl
```

## List models

```powershell
python tools\list_openrouter_models.py --contains gpt --limit 20
python tools\list_openrouter_models.py --contains gemini --limit 20
```

## Notes

- Do not commit `.env`.
- Generated `data/teacher/*.jsonl` should stay local until quality is reviewed.
- If a model fails schema support, use another model or keep `--require-parameters` enabled.
