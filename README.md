# Project Overview

This project involves two key tasks:

1. **Sales Profit Prediction**: Building and evaluating various regression models to predict sales profit.
2. **Image Classification**: Developing a convolutional neural network (CNN) for classifying CIFAR-10 images.

## Data

### Sales Profit Prediction
- **Dataset**: Includes features such as sales, quantity, discount, and profit.
- **Preprocessing**: Handled missing values, outliers, and categorical variable encoding.

### Image Classification
- **Dataset**: CIFAR-10 with 60,000 32x32 color images across 10 classes.

## Exploratory Data Analysis (EDA)

### Sales Profit Prediction
- **Histograms**:
  - **Sales**: Right-skewed distribution with most values at the lower end.
  - **Quantity**: Shows peaks, indicating common order quantities.
  - **Discount**: Most transactions have a 0% discount.
  - **Profit**: Centered around zero with extreme values indicating a mix of highly profitable and unprofitable transactions.
- **Time Series Analysis**:
  - **Sales**: Highly volatile with several peaks.
  - **Profit**: Follows a subdued pattern compared to sales.
- **Correlation Analysis**:
  - **Sales and Profit**: Positive correlation (0.17).
  - **Discount and Profit**: Negative correlation (-0.54).
  - **Quantity**: Weak correlation with Sales (0.21) and Discount (0.12).

## Data Preprocessing

- **Sales Profit Prediction**:
  - Handled missing values and outliers.
  - Encoded categorical variables.
  - Normalized continuous variables.
- **Image Classification**:
  - Resized images to 32x32 pixels.
  - Normalized image channels.

## Model Development

### Sales Profit Prediction
1. **Data Loading and Preprocessing**:
   - Loaded and standardized data.
   - Created new features for non-linear relationships.
2. **Data Splitting**:
   - Split into training (80%) and test sets (20%).
3. **Model Selection and Training**:
   - Algorithms: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor.
   - Trained models on the training set.
4. **Hyperparameter Tuning**:
   - XGBoost optimized using RandomizedSearchCV.
5. **Stacking Regressor**:
   - Combined multiple models for enhanced performance.
6. **Evaluation Metrics**:
   - Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².

### Image Classification
- **Model Architecture**: ImprovedCNN with convolutional layers, dropout layers, and fully connected layers.
- **Training**:
  - Loss Function: Cross-Entropy Loss.
  - Optimizer: Adam.
  - Evaluated accuracy after each epoch.

## Results

### Sales Profit Prediction
- **Best Model**: XGBoost Regressor.
- **Performance**:
  - **Linear Regression**: MSE: 3370.98, MAE: 31.46, R²: 0.8116.
  - **Random Forest Regressor**: MSE: 4205.27, MAE: 25.83, R²: 0.7649.
  - **Gradient Boosting Regressor**: MSE: 2514.96, MAE: 26.54, R²: 0.8594.
  - **XGBoost Regressor**: MSE: 2502.63, MAE: 26.77, R²: 0.8601.
  - **Stacking Regressor**: MSE: 3129.93, MAE: 25.97, R²: 0.8250.

### Image Classification
- **Results**: Improved accuracy after addressing label handling issues.

## Insights and Improvements

### Sales Profit Prediction
- **Models**: Gradient Boosting and XGBoost performed best.
- **Hyperparameter Tuning**: Enhanced XGBoost performance.
- **Model Stacking**: Effective but did not surpass the best individual model.
- **Visualizations**: Identified areas for improvement.

### Image Classification
- **Data Integrity**: Ensured proper preprocessing and label handling.

## Future Work

### Sales Profit Prediction
- Explore further feature engineering and model tuning.
- Investigate alternative stacking approaches.

### Image Classification
- Experiment with different architectures and hyperparameters.
- Explore data augmentation techniques.

## Dependencies

### Sales Profit Prediction
- Libraries: `pandas`, `NumPy`, `matplotlib`, `scikit-learn`, `XGBoost`.


