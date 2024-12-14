# **Heart Disease Prediction using Deep Neural Networks**

## Overview
This project aims to predict the presence of heart disease in individuals using a Deep Neural Network (DNN) model. Heart disease is a leading cause of mortality worldwide, and early detection is crucial for improving patient outcomes. By leveraging predictive analytics, healthcare providers can identify high-risk patients early and provide timely interventions, thus improving healthcare management and reducing costs.

## Problem Statement
Develop a binary classification model to predict the presence or absence of heart disease based on patient medical and personal data. The dataset contains features such as age, sex, cholesterol levels, and more, along with a target variable indicating the presence or absence of heart disease.

## Dataset
The dataset used for this project is publicly available and contains the following features:

### Features:
- **Age**: Age of the patient
- **Sex**: Gender of the patient
- **Chest pain type**: 4 types of chest pain
- **Resting blood pressure**: Patient's blood pressure at rest
- **Serum cholesterol**: Cholesterol level in mg/dl
- **Fasting blood sugar**: Indicator if blood sugar > 120 mg/dl
- **Resting ECG results**: Results of electrocardiographic tests (0, 1, or 2)
- **Max heart rate**: Maximum heart rate achieved
- **Exercise-induced angina**: Whether exercise caused angina
- **Oldpeak**: ST depression induced by exercise
- **Slope**: The slope of the peak exercise ST segment
- **Number of major vessels**: Colored by fluoroscopy (0-3)
- **Thalassemia**: Normal, fixed defect, or reversible defect

### Target:
- **Target**: 1 indicates heart disease is present, 0 indicates it is absent.

## Project Workflow

### 1. Data Exploration and Preprocessing
- Load the dataset and inspect it using pandas.
- Perform exploratory data analysis (EDA) to visualize class distributions and feature distributions.
- Handle missing values by dropping rows with null entries.
- Scale the numeric features using RobustScaler for normalization.

### 2. Feature Engineering
- Drop constant and unique columns.
- Optionally perform Principal Component Analysis (PCA) for dimensionality reduction and visualization.

### 3. Model Building and Training
#### Models:
- **Artificial Neural Network (ANN)**: A simpler model with two hidden layers.
- **Deep Neural Network (DNN)**: A more complex model with four hidden layers for enhanced performance.

#### Training Process:
- Split the dataset into training and testing sets.
- Use `StandardScaler` to scale features for better model performance.
- Add dropout layers to mitigate overfitting.
- Use early stopping to prevent over-training.
- Train models using the Adam optimizer and binary cross-entropy loss function.

### 4. Model Evaluation
- Evaluate the model on the test dataset for accuracy and loss.
- Visualize training and validation metrics over epochs.

### 5. Model Saving
- Save the trained model as `heart_dnn_model.h5` for future use.

## Results
- **ANN Model**: Achieved a test accuracy of **91.22%** and a test loss of **0.1710**.
- **DNN Model**: Achieved a test accuracy of **98.05%** and a test loss of **0.0638**.

## Installation and Usage
### Prerequisites
- Python 3.7 or higher
- Required Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`

### Steps to Run the Code
1. Clone the repository or download the script.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
   ```
3. Place the dataset (`heart.csv`) in the appropriate directory.
4. Run the script in a Python environment or Jupyter Notebook.
5. Review the training and evaluation results.

### Example Commands
To train the DNN model, execute:
```python
python heart_disease_dnn.py
```

## Visualization
The script provides visualizations for:
- Distribution of features.
- Training and validation accuracy over epochs.
- Training and validation loss over epochs.

## File Descriptions
- **`heart_disease_dnn.py`**: Main script containing all the code.
- **`heart.csv`**: Dataset file.
- **`heart_dnn_model.h5`**: Saved DNN model.

## Future Work
- Hyperparameter tuning to further optimize the model.
- Incorporating additional features or datasets for enhanced generalizability.
- Deploying the model for real-time predictions using Flask or FastAPI.

## Acknowledgements
- Dataset sourced from UCI Machine Learning Repository.
- Libraries: TensorFlow, Keras, Scikit-learn, Pandas, Matplotlib, and Seaborn.

## Author
Akanksha Sawant

## License
This project is licensed under the MIT License. Feel free to use and modify the code for educational or commercial purposes.
