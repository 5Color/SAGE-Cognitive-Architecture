# SAGE v2.13.1 - OpenRouter Teacher Dataset Generator

## Run

```powershell
pip install -r requirements-teacher.txt
$env:OPENROUTER_API_KEY="sk-or-..."
$env:SAGE_TEACHER_MODEL="openai/gpt-4o-mini"

python tools\generate_seed_questions.py --limit 80
python tools\generate_teacher_dataset_openrouter.py --seeds data\seeds\generated_seed_questions.jsonl --limit 20 --concurrency 3 --temperature 0.3
python tools\validate_teacher_dataset.py --file data\teacher\sage_teacher_examples.jsonl
```

## Commit

Do not commit `.env`.
Generated `data/teacher/*.jsonl` is optional and should be reviewed first.

```powershell
git add tools/generate_teacher_dataset_openrouter.py
git add tools/generate_seed_questions.py
git add tools/validate_teacher_dataset.py
git add tools/list_openrouter_models.py
git add schemas/sage_teacher_example.schema.json
git add data/seeds/seed_questions.jsonl
git add requirements-teacher.txt
git add .env.example
git add docs/SAGE_v2_13_1_OPENROUTER_TEACHER_GENERATOR.md
git add docs/versions/README_v2_13_1.md

git commit -m "Add SAGE v2.13.1 OpenRouter teacher dataset generator"
git tag v2.13.1
```
