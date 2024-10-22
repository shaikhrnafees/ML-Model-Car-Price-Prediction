# Car Selling Price Prediction

<p align="center">
  <img src="https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/gif1.gif" alt="Project GIF" width="42%" style="display:inline-block; vertical-align:top;" /> 
  <img src="https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/logo.avif" alt="Car Selling Price Prediction Logo" width="12.5%" style="display:inline-block; vertical-align:top;" />
</p>

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Data Cleaning Steps](#data-cleaning-steps)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Libraries Used](#libraries-used)
9. [Conclusion](#conclusion)
10. [Future Directions](#future-directions)
11. [Acknowledgments](#acknowledgments)
12. [Project Details](#project-details)

## Project Overview

The task involves predicting car selling prices using the **"cardekho"** dataset, which includes various features such as name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, and seats. The objective is to build a machine learning model that accurately predicts selling prices based on these car features and histories.

### Key Features of the Dataset:
- **Name**: Model and brand of the car.
- **Year**: Manufacturing year of the car.
- **km_driven**: Total kilometers driven by the car.
- **Fuel**: Type of fuel used (e.g., petrol, diesel).
- **Seller Type**: Type of seller (e.g., individual, dealer).
- **Transmission**: Type of transmission (e.g., manual, automatic).
- **Owner**: Number of previous owners.
- **Mileage**: Fuel efficiency of the car.
- **Engine**: Engine size in liters.
- **Max Power**: Maximum power output of the engine.
- **Seats**: Seating capacity of the car.

## Data Source

The dataset used for this project is sourced from [Cardekho](https://www.cardekho.com) and contains detailed information on various car listings.

## Data Cleaning Steps

Data cleaning is a crucial step in preparing the dataset for modeling. The following steps were undertaken:

1. **Handling Missing Values**:
   - Identified columns with missing data by checking for null values.
   - For numerical columns, filled missing values with the **median** to avoid skewing the data, ensuring a more robust estimation.
   - For categorical columns, filled missing values with the **mode** to maintain the dataset's integrity and prevent loss of information.

2. **Correcting Data Types**:
   - Converted columns to appropriate data types (e.g., `year` to integer, `km_driven` to float) to ensure accurate computations and analyses.
   - Checked and corrected any inconsistencies in data formats (e.g., leading/trailing spaces in categorical variables).

3. **Addressing Outliers**:
   - Analyzed numerical features for outliers using box plots, which helped visualize the spread of data and identify extreme values.
   - Removed outliers beyond a specified threshold (e.g., 1.5 times the interquartile range) to enhance model performance by reducing noise in the data.

4. **Data Quality Assurance**:
   - Ensured there are no duplicate entries in the dataset, which could bias the model.
   - Verified the consistency of data formats across relevant features to maintain uniformity.

## Exploratory Data Analysis (EDA)

Conducted EDA to understand the relationships between the features and the target variable, **selling_price**. Key steps included:
- **Visualizations**: Used scatter plots, histograms, and heatmaps to uncover patterns and correlations between features and the target variable.
- **Feature Relationships**: Analyzed how different features impacted selling prices, allowing for insights into which variables were most significant.

![EDA GIF](https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/dataclean.gif)

## Data Preprocessing

1. **Encoding Categorical Variables**:
   - Utilized **label encoding** for ordinal variables (e.g., owner) and **one-hot encoding** for nominal variables (e.g., fuel, seller_type, transmission) to convert categorical data into a numerical format suitable for modeling.

2. **Scaling Numerical Features**:
   - Applied feature scaling (e.g., StandardScaler or MinMaxScaler) to numerical features such as km_driven, mileage, engine, and max_power to standardize the range, improving the model's performance.

## Model Building

After preprocessing, a machine learning model, specifically the **RandomForestRegressor**, was created to predict selling prices. The model-building process included:
- **Train-Test Split**: Divided the dataset into training and testing sets to evaluate model performance effectively.
- **Model Training**: Trained the RandomForestRegressor on the training set, utilizing its ability to handle non-linear relationships and interactions between features.
- **Model Evaluation**: Evaluated the model's performance using metrics like **R-squared**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)** to assess its accuracy.

![Random Forest Visualization](https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/random%20forest.png)

## Hyperparameter Tuning

Hyperparameter tuning was performed to optimize the model’s performance using techniques such as **Grid Search** or **Random Search**. This process involved:
- Testing various hyperparameters (e.g., number of trees, maximum depth) to find the optimal combination that enhances the model’s predictive capabilities.

## Libraries Used

This project utilized several libraries for data analysis and modeling:

- **NumPy**: 
  - Used for numerical computations and handling arrays. Essential for performing mathematical operations on data.

- **Pandas**: 
  - A powerful data manipulation library that facilitated data cleaning, preprocessing, and analysis through DataFrames.

- **Matplotlib**: 
  - A visualization library used to create static, animated, and interactive visualizations in Python. Important for generating plots and charts.

- **Seaborn**: 
  - Built on top of Matplotlib, Seaborn provides a high-level interface for drawing attractive statistical graphics. Used for visualizing distributions and relationships in the data.

- **Scikit-learn**: 
  - A machine learning library that provides tools for model training, evaluation, and tuning. Key for implementing the RandomForestRegressor and hyperparameter tuning techniques.

- **Time**: 
  - Used for handling time-related tasks, such as measuring execution time for model training and evaluation.

## Conclusion

This project successfully demonstrates the application of machine learning techniques to predict car selling prices based on historical data and various car features. The model's accuracy can help stakeholders make informed decisions in the car selling process, enhancing the understanding of pricing strategies and market trends.

## Future Directions

- Further model improvements could include experimenting with other algorithms like **Gradient Boosting** or **XGBoost** to compare performance.
- Incorporating additional external data sources (e.g., economic indicators, regional demand) to enhance model predictions and adapt to changing market conditions.

## Acknowledgments

- [Cardekho](https://www.cardekho.com) for providing the dataset.
- For a complete list of libraries used for data analysis and modeling, [click here](#libraries-used).

## Project Details

For more information about the project, please refer to the [Python script](https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/car_predict.py).
