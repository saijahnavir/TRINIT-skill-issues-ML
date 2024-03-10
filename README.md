# TRINIT-skill-issues-ML-

# Sexual Harassment Awareness Model

## Overview
This repository contains a simple predictive model focused on addressing sexual harassment by leveraging Natural Language Processing (NLP) techniques and a neural network architecture. The primary goal is to streamline the reporting process for victims and enhance the awareness-raising efforts on online platforms and media outlets.

## Background
As sexual harassment gains increased visibility, more individuals are bravely sharing their experiences through online platforms and media outlets. To improve the reporting process and facilitate efficient awareness campaigns, this model aims to automate incident categorization and prioritize severity levels.

## Problem Statement
The current process for reporting incidents of sexual harassment is often burdensome for victims, requiring manual detailing of occurrences and filling out multiple forms. Readers also face challenges in processing detailed narratives, leading to information overload and difficulty in quickly understanding the nature and severity of incidents. Additionally, existing methods for processing reports may lack effectiveness in categorizing and prioritizing incidents based on severity, resulting in delays in interventions.

## Solution
The implemented model uses NLP techniques to analyze textual descriptions of incidents, focusing on categories such as commenting, ogling/facial expressions/staring, and touching/groping. By automating the categorization process, victims can report incidents more efficiently, and readers can quickly grasp the severity of each incident. The model prioritizes incidents based on their severity, enabling timely and targeted interventions.

## Requirements
Ensure that you have the required libraries installed before running the code. You can install them using the following commands:

```python
pip install nltk scikit-learn tensorflow keras lime
```

## Usage

1. **Clone the Repository:**
   ```python
   git clone https://github.com/your-username/sexual-harassment-awareness.git
   cd sexual-harassment-awareness
   ```

2. **Download Dependencies:**
   Before running the code, download the NLTK stopwords and punkt resources:
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

3. **Run the Code:**
   Execute the `ML02.py` script to train the model and make predictions:
   ```python
   python ML02.py
   ```

4. **Check Predictions:**
   Predictions will be saved to a CSV file named `predictions.csv`. You can review the predictions and the original descriptions in this file.

5. **Examine Explanations:**
   Explanations for the first 100 entries will be saved to `explanations.csv`. This file provides insights into the model's decision-making process.

## Files

- `ML02.py`: Python script containing the model training, evaluation, and prediction code.
- `predictions.csv`: CSV file containing the model predictions on the test dataset.
- `explanations.csv`: CSV file containing explanations for the first 100 entries in the test dataset.
