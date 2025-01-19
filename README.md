# 🏡 House Rent Prediction in India

The goal of this project is to predict rent prices for residential properties in India using machine learning techniques. The prediction model takes into account various features like the number of bedrooms, size, locality, and other important factors that influence rent prices. This tool can help both potential tenants and property owners make informed decisions about rental properties.

## 🔗 Demo
You can try the live demo of this project on [Hugging Face Space: House Rent Prediction in India](https://huggingface.co/spaces/Senasu/House_Rent_Prediction_India).

## 📊 Dataset

This project uses a dataset containing information about over 4700 residential properties available for rent in India. The dataset includes several important features:

- **BHK**: The number of bedrooms, hall, and kitchen in the property.
- **Rent**: The rent price of the property.
- **Size**: The size of the property in square feet.
- **Floor**: The floor on which the property is located (e.g., 'Ground out of 2', '3 out of 5').
- **Area Type**: The type of area used for property size calculation (e.g., Super Area, Carpet Area, or Built Area).
- **Area Locality**: The locality in which the property is located.
- **City**: The city where the property is located.
- **Furnishing Status**: The furnishing status of the property (e.g., Furnished, Semi-Furnished, or Unfurnished).
- **Tenant Preferred**: The type of tenant preferred by the property owner or agent.
- **Bathroom**: The number of bathrooms in the property.
- **Point of Contact**: Contact information for the point of contact regarding the property.

You can access the dataset from Kaggle [here](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset).

## 🤖 Model Results

### 1. **Conversational Modeling Results**

The following models were evaluated using various metrics like R-Squared, RMSE, and MAE:

| Model                    | R-Squared | RMSE         | MAE           |
|--------------------------|-----------|--------------|---------------|
| **XGBRegressor**          | 0.745385  | 13165.40     | 6696.44       |
| **Random Forest Classifier** | 0.719723 | 13812.91   | 6634.96       |
| **Gradient Boosting**     | 0.707885  | 14101.62     | 7144.46       |
| **Ridge**                 | 0.666297  | 15072.04     | 8223.35       |
| **DecisionTreeRegressor** | 0.658065  | 15256.81     | 7621.86       |
| **Extra Tree**            | 0.650126  | 15432.91     | 7653.99       |
| **Lasso**                 | 0.648388  | 15471.19     | 7760.33       |
| **Linear**                | 0.572725  | 17054.76     | 9188.23       |
| **KNeighborRegressor**    | 0.491327  | 18608.50     | 9940.95       |
| **ElasticNet**            | 0.478290  | 18845.44     | 10447.35      |

As we can see, **XGBRegressor** provided the best performance, with the highest R-squared value (0.745) and the lowest RMSE and MAE.

The most important features contributing to the model were:
- **City** (specifically, **city_dumbai**)
- **Bathroom**

### 2. **Deep Learning Model Results**

For the deep learning model, the following results were achieved:

- **R-Squared**: 0.6729
- **RMSE**: 14922.59

Although the deep learning model did not outperform the best conversational models, it still demonstrated reasonable predictive accuracy.

