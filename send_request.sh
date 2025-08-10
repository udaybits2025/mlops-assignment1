#!/bin/bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "medinc": 112.3252,
    "house_age": 41.0,
    "ave_rooms": 6.9841,
    "ave_bedrms": 1.0238,
    "population": 322.0,
    "ave_occup": 2.5556,
    "latitude": 37.88,
    "longitude": -122.23,
    "households": 126.0,
    "rooms_per_household": 5.5,
    "bedrooms_per_room": 0.15,
    "population_per_household": 2.5,
    "medinc_log": 2.12
  }'
