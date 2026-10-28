# StudentInsight — Streamlit Web Application

A standalone dashboard for a Data Mining / KDD mini-project. It applies K-Means clustering to attendance, assignment scores, and mid-semester marks.

## Features
- SQLite demo database with 200 synthetic records, created automatically on first run
- CSV upload for other datasets
- K-Means clustering (K=2–6), feature standardisation, and silhouette score
- Interactive Plotly scatter plot, performance-group filter, and student-ID search
- Cluster summary and CSV download
- KDD workflow explanation inside the app

## Run locally
1. Install Python 3.10 or newer and extract this folder.
2. Open a terminal in this folder.
3. Optional virtual environment:
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`
6. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## CSV columns
Required numeric columns: `attendance`, `assignment_score`, `midsem_marks`. Optional ID column: `student_id`. Rows missing a required feature are excluded.

## Important
The built-in data is synthetic and must not be described as real student data. If using real data, remove personally identifying details and make sure you have permission to use it.

## Viva explanation
StudentInsight follows the KDD process: selection, preprocessing, transformation, data mining, evaluation, and presentation. StandardScaler is used before K-Means because the features have different numeric scales. The silhouette score indicates how separated clusters are, but does not by itself prove the groups are meaningful.
