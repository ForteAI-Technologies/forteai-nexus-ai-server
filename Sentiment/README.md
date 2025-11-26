Sentiment Agent with Vector Memory

Overview
- Adds a lightweight vector DB (Chroma) to persist a style guide and exemplars.
- Ensures responses keep a consistent tone and structure.

Setup
1) Create/set your Google API key env var (same key used by the Node and Python parts if desired):
   - Windows PowerShell:
     $Env:GOOGLE_API_KEY = "<your-key>"

2) Install Python dependencies for this agent:
   pip install -r requirements.txt

Seed style memory (optional but recommended)
- Run once to ensure the base style docs are stored. You can also ingest extra .txt exemplars by setting STYLE_EXEMPLARS_DIR.
   python seed_style_memory.py
   # or
   $Env:STYLE_EXEMPLARS_DIR = "c:\\path\\to\\exemplars"; python seed_style_memory.py

Use in code
- The agent automatically upserts the base style and retrieves relevant snippets for each call.
- Generated outputs are also saved back as exemplars to reinforce consistency over time.

Where else to use the vector DB
- Manager/HR dashboards: store org-specific policy language and retrieve to align recommendations.
- Performance agent: persist rubric definitions and example reviews for consistent scoring.
- Potential agent: persist competency models and exemplar assessments to standardize outputs.
- FAQ/chat: store company HR policy Q&A for quick retrieval-augmented answers.
- Feedback summarization: keep summaries by team/quarter to provide trend-aware context.

Notes
- Data is stored under ./vector_store/sentiment for this agent.
- To clear memory, delete that folder.
- If you move files, ensure imports still work (package vs script).
