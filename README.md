# House Price Prediction System

This project implements an end-to-end machine learning pipeline for predicting house prices in Pakistan using the Pakistan House Price Dataset.

## Project Structure

```
.
├── data/                      # Dataset storage
├── model/                     # Trained model storage
├── src/                       # Source code
│   └── train.py              # Model training script
├── app.py                     # Flask web application
├── params.yaml                # Model and training parameters
├── dvc.yaml                   # DVC pipeline definition
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Initialize DVC:
   ```bash
   dvc init
   ```

3. Train the model:
   ```bash
   python src/train.py
   ```

4. Run the Flask application:
   ```bash
   python app.py
   ```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make price predictions

Example prediction request:
```json
{
  "area": 2000,
  "bedrooms": 4,
  "baths": 3,
  "city": "Karachi",
  "province_name": "Sindh"
}
```

## DVC Pipeline

The project uses DVC to manage the ML pipeline. To reproduce the pipeline:

```bash
dvc repro
```

## Configuration

Model parameters can be adjusted in `params.yaml`.