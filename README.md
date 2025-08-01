# air-pm10-forecast
Air-PM10-Forecast is a Docker-packaged system designed for the AIR-PPM Hackathon 2025, capable of generating accurate 24-hour hourly PM₁₀ forecasts based on historical air quality and weather data.

# Docker scripts
### Build
```console
docker build -t air-pm10-forecast .
```
### Run prediction only
```console
docker run --rm -v "/absolute/path/to/air-pm10-forecast/data:/data" air-pm10-forecast --data-file /data/input.json --output-file /data/output.json
```
### Run prediction only (in constraint environment)
```console
docker run --rm --cpus="1.0" --memory="2g" -v "/absolute/path/to/air-pm10-forecast/data:/data" air-pm10-forecast --data-file /data/input.json --output-file /data/output.json
```
### Run retrain
```console
docker run --rm -v "/absolute/path/to/air-pm10-forecast/data:/data" air-pm10-forecast --data-file /data/training_input.json --output-file /data/output.json --retrain
```