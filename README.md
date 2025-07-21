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
    cd airfoil-noise-prediction
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the entire pipeline (data download, training, evaluation, and demo prediction), execute the main script:

```bash
python airfoil_noise_prediction.py
```

The script will print the model's performance metrics and feature importances to the console.

#### Predicting on New Data

To predict the sound pressure level for new airfoil data:

1. Prepare a CSV file named `demo_new_data.csv` in the project directory with the following columns (no header for `sound_pressure_level`):

    ```
    frequency,angle_of_attack,chord_length,free_stream_velocity,suction_side_thickness
    ```

    Example:
    ```
    frequency,angle_of_attack,chord_length,free_stream_velocity,suction_side_thickness
    1250,0.5,0.3048,71.3,0.00266337
    250,2.0,0.2032,50.0,0.0015
    1000,5.0,0.1016,40.0,0.0020
    3150,7.5,0.2540,60.0,0.0030
    500,10.0,0.1524,55.0,0.0022
    ```

2. Run the script. It will generate an Excel file named `demo_predictions.xlsx` with two sheets:
    - **InputData**: The original input data.
    - **Predictions**: The input data plus predicted `sound_pressure_level` and a `prediction_stddev` column indicating the model's uncertainty (lower values mean higher confidence).

---

## 5. Project Structure

```
.
├── airfoil_noise_prediction.py  # Main script for data loading, training, evaluation, and prediction.
├── demo_new_data.csv           # Example input data for prediction.
├── demo_predictions.xlsx       # Output Excel file with predictions and confidence.
├── requirements.txt            # Required Python libraries.
└── README.md                   # This file.
```
