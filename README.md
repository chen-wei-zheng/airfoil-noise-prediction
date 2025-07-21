
# AI-Powered Airfoil Self-Noise Prediction

**Author:** Chenwei.Zheng + AI
**Date:** July 2025

---

## 1. Project Overview

This project demonstrates the use of machine learning to predict the scaled sound pressure level (self-noise) of an airfoil based on its aerodynamic characteristics. The model is trained on the **NASA Airfoil Self-Noise dataset** from the UCI Machine Learning Repository.

The primary goal is to build a highly accurate regression model that can serve as a surrogate for complex physical simulations or expensive experiments, allowing for rapid noise estimation.

---

## 2. Key Results & Performance

A `RandomForestRegressor` model was trained and evaluated, yielding excellent performance on the unseen test data.

*   **R-squared (R²) Score:** **0.935**
    *   *This indicates that the model successfully explains 93.5% of the variance in the airfoil noise data, demonstrating a very strong fit.*
*   **Mean Absolute Error (MAE):** **1.31 dB**
    *   *On average, the model's predictions are off by only 1.31 decibels, showing high precision.*

---

## 3. Key Insights: What Drives Airfoil Noise?

One of the most valuable outputs of the model is the feature importance analysis, which identifies the primary factors contributing to the noise level.

| Feature                  | Importance |
| ------------------------ | ---------- |
| **Frequency**              | 41.6%      |
| **Suction Side Thickness** | 40.6%      |
| Chord Length             | 9.3%       |
| Angle of Attack          | 4.2%       |
| Free-stream Velocity     | 4.2%       |

This analysis reveals that **frequency** and the **suction side displacement thickness** are the dominant drivers of self-noise in this dataset, accounting for over 82% of the predictive power.

---

## 4. How to Run This Project

These instructions will guide you through setting up the environment and running the prediction script.

### Prerequisites

*   Python 3.7+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chen-wei-zheng/airfoil-noise-prediction
    cd main
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the entire pipeline (data download, training, and evaluation), execute the main script:

```bash
python airfoil_noise_prediction.py
```

The script will print the model's performance metrics and feature importances to the console.

---

## 5. Project Structure

```
.
├── airfoil_noise_prediction.py  # Main script for data loading, training, and evaluation.
├── requirements.txt             # Required Python libraries.
└── README.md                    # This file.
```
