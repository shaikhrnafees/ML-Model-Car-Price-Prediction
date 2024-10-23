# Car Selling Price Prediction

## Project Overview

The task involves predicting car selling prices using the **"cardekho"** dataset, which includes various features such as name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, and seats. The objective is to build a machine learning model that accurately predicts selling prices based on these car features and histories.

<p align="center">
  <img src="https://images.yourstory.com/cs/wordpress/2014/08/CarDekho_FeaturedImage_YS.png?mode=crop&crop=faces&ar=2%3A1&format=auto&w=1920&q=75" alt="Car Selling Price Prediction Logo" width="52.5%" />
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

## Data Source

The data used for this project is sourced from the [Cardekho](https://www.cardekho.com) website, which provides information on various car features and histories.

<p align="center">
  <img src="https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/c8684078684029.5cac4a000c7e1.gif" alt="Project GIF" width="40%" />
</p>

## Data Cleaning Steps

Data cleaning involved:
- Handling missing values.
- Correcting data types.
- Addressing outliers.
- Standardizing feature formats, such as converting all mileage units to km/l.

<p align="center">
  <img src="https://www.business2community.com/wp-content/uploads/2020/01/analyze-gif.gif" alt="Data Cleaning GIF" width="40%" />
</p>

## Exploratory Data Analysis (EDA)

The EDA involved:
- Analyzing relationships between features and the target variable (selling price).
- Using visualizations to identify patterns, such as correlations between mileage, year, and price.
- Exploring distributions of categorical variables like fuel type and transmission.

## Data Preprocessing

Steps in data preprocessing:
- Encoding categorical variables (fuel, seller_type, transmission, owner).
- Scaling numerical features (km_driven, mileage, engine, max_power) using standard scaling techniques.
- Splitting the data into training and testing sets.

## Model Building

- A **RandomForestRegressor** was used to build the predictive model.
- The model was trained using the processed dataset, and its performance was evaluated with metrics like R-squared and Mean Squared Error.

<p align="center">
  <img src="https://i0.wp.com/innovationyourself.com/wp-content/uploads/2023/10/Random-Forest-Regression.png?fit=1200%2C600&ssl=1" alt="Random Forest Model" width="50%" />
</p>

## Hyperparameter Tuning

Hyperparameter tuning was performed to optimize model performance, adjusting parameters such as:
- Number of estimators.
- Maximum depth of trees.
- Minimum samples split.

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Matplotlib**: For visualizations.
- **Seaborn**: For advanced plotting.
- **Scikit-Learn**: For machine learning model building and evaluation.

## Conclusion

The project successfully built a predictive model for car selling prices, providing insights into key factors that influence pricing. The model can be further refined for even better performance.

## Future Directions

- Implement more complex models like **Gradient Boosting** or **XGBoost**.
- Use more advanced feature engineering techniques to capture non-linear relationships.
- Integrate additional data sources for richer insights.

## Acknowledgments

- Thanks to [Cardekho](https://www.cardekho.com) for the dataset.
- For more information on libraries used, [click here](#libraries-used).

## Project Details

For more information about the project, please refer to the [Python script](https://github.com/shaikhrnafees/ML-Model-Car-Price-Prediction/blob/main/car_predict.py).
