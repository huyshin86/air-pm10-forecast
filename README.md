# air-pm10-forecast
Air-PM10-Forecast is a Docker-packaged system designed for the AIR-PPM Hackathon 2025, capable of generating accurate 24-hour hourly PM₁₀ forecasts based on historical air quality and weather data.

# 📂 Structure
```                                      
├── Dockerfile
├── README.md
├── requirements.txt
├── run_model.py                    # 🌟 Main entry point for Docker
├── data/
│   ├── raw/                       
│   ├── processed/                 
│   └── sample/                    # 🧪 Sample data.json for testing
├── notebooks/                     # 🧑‍💻 Development & experimentation
│   ├── data_engineering/
│   ├── feature_engineering/       # EDA + Feature prototypes only
│   └── modeling/
├── src/                           # 🏗️ Submission code (production only)
│   ├── io_utils.py                # Load data.json / Save output.json
│   ├── predictor.py               # Core model logic + inline feature 
│   └── landuse.py                 # (Optional) Landuse extractor if needed
├── test/
```