Heart Disease Prediction Web Application

An AI-powered web application that predicts the risk of heart disease using Machine Learning.
The project uses a trained Random Forest model saved as best_rf_model.pkl, developed using
Jupyter Notebook and deployed as an interactive Streamlit web application.

--------------------------------------------------

Project Overview

Heart disease is one of the leading causes of death worldwide.
This project helps in early prediction of heart disease risk by analyzing
patient medical parameters and providing probability-based results.
The application is designed as a professional internship-level project.

--------------------------------------------------

Technologies Used

Python
Pandas
NumPy
Scikit-learn
Jupyter Notebook
Streamlit
Plotly
Git & GitHub

--------------------------------------------------

Dataset Information

The dataset (heart.csv) contains medical attributes such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- ECG results
- Maximum heart rate
- Exercise-induced angina

Target column:
- num (indicates presence of heart disease)

--------------------------------------------------

Model Training (Jupyter Notebook)

Model training is performed in Jupyter Notebook (heartdisease.ipynb).

Steps included:
- Data loading and preprocessing
- Handling missing values
- Encoding categorical features
- Feature selection
- Random Forest model training
- Model evaluation
- Saving the trained model as best_rf_model.pkl

The saved model file (best_rf_model.pkl) is included in this repository
and is directly loaded by the Streamlit application for prediction.

--------------------------------------------------

Trained Model File

best_rf_model.pkl

This file contains the trained Random Forest machine learning model.
It is generated in the Jupyter Notebook and used in heart.py for
real-time heart disease prediction.

--------------------------------------------------

Web Application (Streamlit)

The trained model is deployed using Streamlit (heart.py).

Features of the web application:
- Dark themed professional user interface
- Patient data input form
- Heart disease risk prediction
- Animated risk probability gauge
- Interactive charts and graphs
- Model feature importance visualization

--------------------------------------------------

Project Structure

heart-disease-streamlit/
|
|-- heart.py
|-- heartdisease.ipynb
|-- heart.csv
|-- best_rf_model.pkl
|-- requirements.txt
|-- README.md

--------------------------------------------------

How to Run the Project Locally

Step 1: Clone the repository
git clone https://github.com/yourusername/heart-disease-streamlit.git
cd heart-disease-streamlit

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the Streamlit application
streamlit run heart.py

The application will open in your browser at:
http://localhost:8501

--------------------------------------------------

Why requirements.txt is Important

The requirements.txt file contains all required Python libraries
needed to run this project. It ensures environment consistency
and is required for deployment and collaboration.

It is generated using:
pip freeze > requirements.txt

--------------------------------------------------

Deployment

The project is hosted on GitHub and can be deployed on Streamlit Cloud
to obtain a live web application URL.

--------------------------------------------------

Author

Rishika Peddireddy
Graduated
Machine Learning & Data Analytics

--------------------------------------------------

License

This project is created for internship at DynamicNetwork.
