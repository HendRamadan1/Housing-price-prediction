# Housing price prediction

## Overview 
This Data science project analyses the [House Prices - Advanced Regression Techniques | Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), predicting the `SalePrice` based on all other features in the dataset
**Technologies used:**
- `python 3.12`
- `matplotlib`
- `pandas`
- `numpy`
- `seaborn`
- `XGBRegressor`
- `Sklearn`

## Step 1: Exploratory Data Analysis 

- Know the null and unique values in each column in the dataset 
- Differentiating between object and numeric values for their corresponding suitable data cleaning technique 
- Plotted the numeric data correlation with the target column `SalePrice`
- Plotted a frequency histogram for each numeric column in the dataset
- Created a Correlation `Heatmap` 
- Created a function to find all null values in a column, which was useful in the long run
```python
def getMissingValues(df):
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    missing_values = missing_values / len(df)
    return [missing_values], missing_values.__len__()
```

- Created a function to find all unique values in the dataset
```python
def getUniqueVaules(df):

    unique_values = []
    for i in df.columns:
        unique_values.append((i , df[i].nunique()))
    return f" {unique_values} "
```

- Plotted a frequency histogram for each categorical column, and filled the null values with suitable 
- Handled numeric null records with `Measure of Center` techniques (mean, median, mode)
- Export cleaned data to new files

--- 
## Step 2: Feature Engineering
- Added the following columns that are derived from other columns:
```python 
train['TotalArea'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalArea'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

# Age of the house when sold
train['HouseAge'] = train['YrSold'] - train['YearBuilt']

test['HouseAge'] = test['YrSold'] - test['YearBuilt']

# Interaction term: Overall quality * Living area
train['QualLivArea'] = train['OverallQual'] * train['GrLivArea']

test['QualLivArea'] = test['OverallQual'] * test['GrLivArea']
```

- used the `log transform` to convert skewed data into an acceptable domain for training
```python
skewed_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF']

  

for col in skewed_features:
    train[col] = np.log1p(train[col])
    test[col] = np.log1p(test[col])

  

# Log-transform the target variable
y_train = np.log1p(y_train)  # Reverse later with np.expm1()
```

- used `Ordinal encoding` for some columns 
```python 
# Define ordinal mappings (customize based on your data)

ordinal_mappings = {

    'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},

    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},

    'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}

}

  

for col, mapping in ordinal_mappings.items():

    train[col] = train[col].map(mapping)

    test[col] = test[col].map(mapping).fillna(0)  # Handle missing/unseen categories in test
```

- Saved the processed data for the next Stage: *Model building*

---

Here’s the revised version with **inline equations** using `$...$` for GitHub compatibility:

---

# XGBRegressor Functionality

XGBRegressor is a gradient-boosted decision tree (GBDT) algorithm for regression. Below are its core components and equations:

---

## 1. **Objective Function**
The regularized objective combines loss and tree complexity:  
$\text{Obj}(\theta)=\sum_{i=1}^n L(y_i,\hat{y_i})+ \ sum_{k=1}^K\Omega(f_k)$  
- **Loss Term**: For regression, the squared error is commonly used:  
  $L(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2$  
- **Regularization Term**: Penalizes tree complexity:  
  $\Omega(f_k) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$  
  - $T$: Number of leaves  
  - $w_j$: Weight of leaf $j$  
  - $\gamma, \lambda, \alpha$: Regularization hyperparameters.

---

## 2. **Additive Training**
Predictions are updated iteratively at each boosting step $t$:  
$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$  
- $\eta$: Learning rate (shrinkage factor)  
- $f_t$: Weak learner (tree) added at step $t$.

---

## 3. **Taylor Approximation**
The loss is approximated using gradients ($ g_i $) and hessians ($ h_i $):  
$\text{Obj}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $  
- **Gradients** (1st-order derivative for squared error):  
  $g_i = \frac{\partial L}{\partial \hat{y}^{(t-1)}} = \hat{y}^{(t-1)} - y_i$  
- **Hessians** (2nd-order derivative for squared error):  
  $h_i = \frac{\partial^2 L}{\partial (\hat{y}^{(t-1)})^2} = 1$  

---

## 4. **Optimal Leaf Weight**
For leaf $j$ with instance set $I_j$:  
$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$

---

## 5. **Split Gain Formula**
The gain for splitting a node into left ($ L $) and right ($ R $) subsets:  
$\text{Gain} = \frac{1}{2} \left( \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma$  
- $G_L = \sum_{i \in I_L} g_i$, $G_R = \sum_{i \in I_R} g_i$  
- $H_L = \sum_{i \in I_L} h_i$, $H_R = \sum_{i \in I_R} h_i$  

---

## 6. **Key Hyperparameters**
| Parameter       | Math Symbol | Description                          |
|-----------------|-------------|--------------------------------------|
| `learning_rate` | $\eta$    | Shrinks tree contributions           |
| `gamma`         | $\gamma$  | Minimum loss reduction for a split   |
| `lambda`        | $\lambda$ | L2 regularization on leaf weights   |
| `alpha`         | $\alpha$  | L1 regularization on leaf weights   |
| `max_depth`     | -           | Maximum depth of a tree              |
| `subsample`     | -           | Fraction of samples used per tree    |

---

## Summary
XGBRegressor builds trees greedily by:  
1. Approximating the loss with Taylor expansion,  
2. Calculating optimal leaf weights,  
3. Selecting splits that maximize gain.  

Equations use inline notation for GitHub Markdown compatibility.
