# Hotel Booking Cancellation Prediction

## Overview

This project aims to develop a predictive model that forecasts the likelihood of hotel booking cancellations. By analyzing historical booking data, the model will help hotel management understand customer behavior, optimize booking strategies, and improve operational efficiency.

The dataset used contains booking details and customer information, which will be processed and analyzed to identify patterns contributing to cancellations.

## Dataset

The dataset can be accessed from Kaggle: [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

## Objective

- Build a predictive model to determine the likelihood of hotel booking cancellations.
- Identify key factors that contribute to cancellations and help hotel management make data-driven decisions.

## Assumptions

1. **Sufficient Historical Data**: The dataset is assumed to be representative of future booking patterns.
2. **Independent Observations**: All observations in the dataset are considered independent of one another.
3. **Customer Behavior Stability**: It is assumed that customer behavior remains consistent over time.
4. **Removal of Personally Identifying Data**: The dataset has no personally identifiable information (PII).
5. **Overlapping Features**: The `distribution_channel` feature overlaps significantly with `market_segment` and is therefore removed.
6. **Data Leakage**: The `reservation_status` and `reservation_status_date` columns are highly correlated with the target variable (`is_cancelled`) and are removed to prevent data leakage.
7. **Feature Complexity**: Unnecessary features that may degrade model performance or add complexity are dropped.

## Limitations of the Dataset

- **Imbalanced Dataset**: The target variable (`is_cancelled`) is imbalanced, with 75,166 non-canceled bookings and 44,224 canceled bookings.
- **Missing Data**: The `company` column has too many missing values to be useful.

## Design Decisions

### 1. Data Ingestion
- The dataset is fetched from an Amazon S3 bucket.

### 2. Libraries Used
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Statistical data visualization.
- **Scikit-learn**: Machine learning algorithms and tools.

### 3. Exploratory Data Analysis (EDA)
- **Sweetviz Profiling**: Automated data analysis and visualization.
- **Y-data Profiling**: Comprehensive EDA and feature analysis.
- **Correlation Analysis**: Understanding the relationship between features and the target variable.
- **Pair-plots**: Visualization of feature interactions.

### 4. Data Preprocessing
- Separate numerical and categorical features.
- Handle missing values.
- Label encoding for categorical variables.
- Perform feature selection to retain only important features.

### 5. Model Selection
Multiple models are trained and evaluated:
- **Logistic Regression**: A simple and interpretable model.
- **Random Forest**: A powerful ensemble learning model.
- **XGBoost Classifier**: Gradient boosting model for improved performance.
- **Artificial Neural Networks (ANN)**: To capture complex patterns.

### 6. Model Deployment
The final model is deployed using **AWS SageMaker**, which simplifies the process of building, training, and deploying machine learning models at scale.

## How to Use

1. Clone this repository:
    ```bash
    git clone https://github.com/ajsuraj/Hotel-Booking-Cancellation-Prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Fetch the dataset from the Kaggle link and place it in the `data` folder.
4. Run the notebook or Python scripts to preprocess the data and train the model:
    ```bash
    python train_model.py
    ```
5. Deploy the model using AWS SageMaker for production-level use.

## Future Work

- **Address Class Imbalance**: Implement techniques like oversampling, undersampling, or SMOTE to handle the imbalanced target variable.
- **Hyperparameter Tuning**: Further fine-tune model hyperparameters to optimize performance.
- **Feature Engineering**: Explore advanced feature engineering techniques to improve prediction accuracy.

## Conclusion

This project provides valuable insights into hotel booking behavior and offers a model that can predict cancellations, assisting hotel managers in making data-driven decisions to improve operational efficiency and customer satisfaction.

## Contact

For any questions or suggestions, please feel free to reach out to me at [Email](suraj.dataml@gmail.com) .

