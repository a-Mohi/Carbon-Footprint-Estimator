# Carbon Footprint Estimator


🎯 SDG Goal: SDG 13 – Climate Action

This project focuses on predicting CO₂ emissions of vehicles using Machine Learning techniques. It aligns with United Nations Sustainable Development Goal 13 (Climate Action) by promoting awareness of vehicle carbon emissions and enabling data-driven environmental decisions.


---

📌 Project Overview

Transportation is a major contributor to greenhouse gas emissions. This project builds a Multiple Linear Regression (MLR) model to estimate vehicle CO₂ emissions based on:

Engine Displacement (Engine Size)

Number of Cylinders

Fuel Type (Petrol/Diesel – One Hot Encoded)


Additionally:

Outlier detection is performed on engine size to improve linear relationship.

Model performance is evaluated using regression metrics.



---

🎯 Objectives

Predict CO₂ emissions using vehicle specifications

Apply One Hot Encoding for categorical fuel types

Perform outlier detection on engine size

Improve model performance and linearity

Evaluate the regression model using standard metrics



---

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn



---

📊 Dataset Features

Feature	Description

Engine Size	Engine displacement (in liters)
Cylinders	Number of engine cylinders
Fuel Type	Petrol / Diesel (Categorical)
CO2 Emissions	Target variable (g/km)



---

🔎 Data Preprocessing Steps

1. Data Cleaning


2. Handling Missing Values


3. One Hot Encoding for Fuel Type


4. Outlier Detection on Engine Size (IQR Method / Z-Score)


5. Feature Scaling (if applied)


6. Train-Test Split




---

🤖 Model Used

Multiple Linear Regression (MLR)

The mathematical form of the model:

CO₂ = β₀ + β₁(EngineSize) + β₂(Cylinders) + β₃(FuelType) + ε

Where:

β₀ = Intercept

β₁, β₂, β₃ = Coefficients

ε = Error term



---

📈 Model Evaluation Metrics

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score



---

🚀 Results

Established a positive correlation between engine size and CO₂ emissions

Outlier removal improved linear fit

Model achieved good R² performance indicating strong predictive capability



---

🌱 Impact – SDG 13 (Climate Action)

This project contributes toward United Nations SDG 13 by:

Raising awareness about vehicle emissions

Supporting data-driven climate decisions

Encouraging sustainable vehicle choices

