# 🐧 Penguin Species Predictor

This repository contains a Python machine learning model that predicts the species of a penguin based on the following features:

- **Island**
- **Sex**
- **Bill Length (mm)**
- **Bill Depth (mm)**
- **Flipper Length (mm)**
- **Body Mass (g)**

The model is trained using the [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins), a popular alternative to the Iris dataset for classification tasks.

## 🔍 Features

- Input: Structured data (sex, island, and body measurements)
- Output: One of three penguin species — **Adelie**, **Chinstrap**, or **Gentoo**
- Preprocessing includes:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling (if applicable)
- Model serialized using `joblib` for easy reuse

## 🚀 Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/penguin-species-predictor.git
   cd penguin-species-predictor
