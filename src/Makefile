

make run:
	uv run python main.py
app:
	uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000

ingect:
	uv run python -m scripts.ingest_vezeeta --batch-size 64 --reset

ingect_default:
	uv run python -m scripts.ingest_vezeeta