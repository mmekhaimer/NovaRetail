# NovaRetail Customer Intelligence Dashboard

## Purpose
An interactive executive dashboard that translates NovaRetail's transaction-level
data into clear, actionable insights on revenue drivers, customer segments,
satisfaction, retention risk, and growth opportunities — built for
Sophia Martinez, Director of Customer Intelligence.

## Business Questions Answered
1. Which customers and customer segments generate the most revenue?
2. Which customer segments appear to be at risk of declining or churning?
3. Where should NovaRetail invest to maximize customer growth and retention?

## Required Files
- `app.py` — the Streamlit application
- `NR_dataset.xlsx` — the transaction dataset (must be in the same folder as `app.py`)
- `requirements.txt` — Python package dependencies

## Local Installation
1. Clone or download this repository.
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
```bash
streamlit run app.py
```

## Deploying to GitHub and Streamlit Community Cloud
1. Create a new GitHub repository.
2. Upload the following files to the repository:
   - `app.py`
   - `NR_dataset.xlsx`
   - `requirements.txt`
   - `README.md`
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
4. Click "New app" and connect your GitHub repository.
5. Select the correct branch.
6. Set `app.py` as the application entry point.
7. Click "Deploy."
8. Streamlit Cloud will automatically install the packages listed in `requirements.txt`.

## Project Structure
```
novaretail-dashboard/
├── app.py
├── NR_dataset.xlsx
├── requirements.txt
└── README.md
```
