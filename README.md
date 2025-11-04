# Credit Score Dashboard

Interactive Streamlit dashboard for credit scoring analysis and client evaluation.

## Overview

User-friendly web application for loan officers to evaluate client credit risk. Displays predictions, feature importance, and client comparisons with visual analytics.

**Dashboard :** https://francklm3-p7-dashboard-deploy-dashboard-67h1jz.streamlitapp.com/
**API :** https://credit-score-api-572900860091.europe-west1.run.app/docs

## Features

- Client credit score evaluation
- Real-time predictions via API
- Local fallback model (for offline mode)
- Feature importance visualization (SHAP)
- Client comparison with global statistics
- Interactive charts and gauges
- Responsive design

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
         ├─── (Primary) ──→ Cloud Run API
         │
         └─── (Fallback) ──→ Local Pipeline
```

**Prediction Flow:**
1. Try API call to Cloud Run (production model)
2. If API fails, use local model as fallback
3. Display results with visualizations

## Project Structure

```
P7_dashboard_deploy/
├── dashboard.py                # Main Streamlit application
├── dashboard_functions.py      # Helper functions for visualization
├── utils.py                    # Utility functions (API calls, data loading)
├── ressource/
│   ├── pipeline                # Legacy preprocessor (fallback)
│   └── classifier              # Legacy model (fallback)
├── data/
│   └── dataset_sample.csv      # Sample client data
├── requirements.txt            # Python dependencies
└── Dockerfile                  # Container for deployment

```

## Local Development

### Installation

```bash
# Clone repository
git clone https://github.com/FranckLM3/P7_dashboard_deploy.git
cd P7_dashboard_deploy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Start dashboard
streamlit run dashboard.py

# Dashboard opens at http://localhost:8501
```

### Configuration

Set API URL via environment variable (optional):

```bash
# Use custom API URL
export CREDIT_SCORE_API_URL="https://your-api-url.run.app"
streamlit run dashboard.py
```

Default API: `https://credit-score-api-572900860091.europe-west1.run.app`

## Usage

1. **Select Client**: Choose client ID from dropdown
2. **View Score**: See credit score with gauge visualization
3. **Analyze Features**: Review feature importance and SHAP values
4. **Compare**: Compare client with similar profiles
5. **Export**: Download analysis results

## API Integration

The dashboard consumes the Credit Score API:

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "id": 162473
}
```

**Response:**
```json
{
  "credit_score": 0.215,
  "advice": "No payment difficulties"
}
```

**Fallback Mode:**
- If API is unavailable, uses local model
- Ensures continuity of service
- Same prediction logic as API

## Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy with `dashboard.py` as main file

### Deploy with Docker

```bash
# Build image
docker build -t credit-dashboard .

# Run container
docker run -p 8501:8501 credit-dashboard
```

## Tech Stack

- **Framework**: Streamlit 1.50.0
- **Visualization**: Plotly
- **ML**: scikit-learn 1.5+, LightGBM 4.1+
- **Data**: Pandas, NumPy
- **API Client**: Requests

## Related Projects

- **API**: [P7_api_deploy](../P7_api_deploy) - FastAPI REST service for predictions
- **ML Training**: [P7_Implementez_modele_scoring](../P7_Implementez_modele_scoring) - Model training pipeline

## Performance

- **Response Time**: < 2s (API) / < 1s (local)
- **Concurrent Users**: Supports multiple simultaneous sessions
- **Data Volume**: 300k+ clients in dataset

## License

OpenClassrooms project - Educational purposes

---

*Last update: November 2025*
