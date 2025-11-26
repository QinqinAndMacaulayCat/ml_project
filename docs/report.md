
# Predicting Corporate Bond YTM Changes Using Machine Learning

Team members: Yuxi Geng, Yutung Lin, Qinqin Huang

## 1. Objective

The objective of this project is to predict corporate bond yield to maturity (YTM) changes using firm-level fundamentals, market-based risk factors, and macroeconomic indicators.  

By applying various machine learning regression models such as ElasticNet, Boosting and Multi-layer Perceptron, we aim to improve prediction accuracy compared to traditional linear regression models.

In addition, we used the change direction of YTM as a classification target since there are many outliers in the YTM change values that may affect regression performance. We applied classification models such as Logistic Regression, ElasticNet Classifier, and Boosting Classifier to predict the direction of YTM changes.

---

## 2. Methodology

### 2.1 Data Description

We will utilize data from Wharton Research Data Services (WRDS) to construct our dataset. The data frequence is monthly and data range is from July 2002 to February 2025.


#### 2.1.1 Target Variable

The target variable is the corporate bond yield to maturity (YTM) change, defined as the difference in YTM over a one-month horizon:

$$
\Delta \text{YTM}_i = \text{YTM}_{i,t} - \text{YTM}_{i,t-1}
$$

where $\text{YTM}_{i,t}$ is the yield to maturity of bond $i$ at the end of month $t$.

For classification, the target variable is defined as:

$$
\text{YTM\_Direction}_i = 
\begin{cases}
up, & \text{if } \Delta \text{YTM}_i > 1e-6 \\
neutral, & \text{if } \Delta \text{YTM}_i \approx 0 \\
down, & \text{if } \Delta \text{YTM}_i < -1e-6
\end{cases}
$$

The Data is from the "WRDS Bond Returns" database, which provides monthly bond yields and other bond characteristics, sourced from TRACE and Mergent FISD. The dataset has been cleaned and more suitable for analysis. 

We only consider senior bonds and exclude defaulted bonds. Additionally, the yields exceeding [-1, 1] are treated as outliers and removed from the dataset.

Total number of observations meeting the criteria is approximately 1760852, covering 74191 unique bonds and 3187 unique firms over the sample period.


#### 2.1.2 Features

1. Key Columns

- Bond Identifier: CUSIP
- Firm Identifier: PERMNO
- Date: End of Month

We have checked the missingness and uniqueness of these key columns.

2. Bond-level features

- Time to Maturity (tmt)
- Coupon Rate (coupon)
- t_spread: the weighted average bid-ask spread of the bond which measures liquidity.
- return_eom: bond return of last month. 
- Credit Ratings: One-hot encoded variables for ratings from AAA to D (rating_A, rating_AA, ..., rating_D). The ratings are weighted averages ratings from WRDS Bond Returns database rather than S&P, Moody's, or Fitch alone.
- Rating Change Indicators: Binary variables indicating whether there was an upgrade or downgrade in the bond's credit rating in the past month (upgrade, downgrade).

Here, we used lagged values of features except time to maturity and coupon rate to avoid look-ahead bias.


| Variable       | Count     | Mean   | Std    | Min    | 25%    | 50%    | 75%    | Max    |
|----------------|-----------|--------|--------|--------|--------|--------|--------|--------|
| `tmt`          | 3.56e+6   | 6.35   | 8.67   | 0.00   | 1.61   | 3.54   | 6.89   | 102.13 |
| `coupon`       | 3.56e+6   | 0.03   | 0.03   | 0.00   | 0.00   | 0.03   | 0.06   | 0.34   |
| `t_spread`     | 1.79e+6   | 0.01   | 0.01   | 0.00   | 0.00   | 0.00   | 0.01   | 2.00   |
| `yield`        | 3.39e+6   | 0.01   | 0.13   | -1.00  | 0.00   | 0.03   | 0.05   | 1.00   |
| `ret_eom`      | 3.42e+6   | 0.01   | 1.18   | -1.00  | 0.00   | 0.00   | 0.01   | 2090.0 |
| `rating_A`     | 3.42e+6   | 0.20   | 0.40   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_AA`    | 3.42e+6   | 0.04   | 0.21   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_AAA`   | 3.42e+6   | 0.01   | 0.09   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_B`     | 3.42e+6   | 0.03   | 0.16   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_BB`    | 3.42e+6   | 0.04   | 0.20   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_BBB`   | 3.42e+6   | 0.23   | 0.42   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_C`     | 3.42e+6   | 0.00   | 0.02   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_CC`    | 3.42e+6   | 0.00   | 0.03   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_CCC`   | 3.42e+6   | 0.01   | 0.09   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `rating_D`     | 3.42e+6   | 0.00   | 0.02   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `upgrade`      | 3.42e+6   | 0.00   | 0.04   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |
| `downgrade`    | 3.42e+6   | 0.00   | 0.05   | 0.00   | 0.00   | 0.00   | 0.00   | 1.00   |

3. Firm-level features
We incorporate firm-level information at both the daily and quarterly frequency.

• Daily Stock Price and Returns:
Includes the daily closing stock price (PRC), daily stock return (RET), and trading volume (VOL). These variables capture short-term market performance and trading activity.

• Daily Trading Liquidity Measures:
Includes the daily bid and ask quotes (BIDLO, ASKHI) and the number of trades (NUMTRD) as indicators of market liquidity.

• Quarterly Accounting Fundamentals:
Includes total assets (atq), total liabilities (ltq), common equity (ceqq), shares outstanding (cshoq), and quarterly net income (niq). These variables summarize the firm’s financial position and profitability at the quarterly reporting date.

• Firm Identification:
The PERMNO identifier is present in both daily stock data and quarterly fundamental data, enabling direct alignment of firm-level information across time.

4. Macroeconomic features (yuxi)

We augment the bond- and firm-level data with monthly macroeconomic and broad market indicators that capture aggregate risk and business-cycle conditions:

• Market Index Return and Risk Sentiment: Includes the monthly excess equity market return (sp500_ret) and the monthly change in implied volatility (vix_chg). These variables capture broad stock market performance and shifts in overall risk appetite.

• Interest Rate Level and Term Structure: Includes the level of the 3-month Treasury yield (gs3m), monthly changes in short- and long-term interest rates (ir3m_chg, ir10y_chg), and the term spread between long and short maturities (term_spread). These variables summarize the stance of monetary policy and the shape of the yield curve.

• Macroeconomic Conditions: Includes real GDP growth (gdp_gr) and CPI inflation (cpi_infl). These variables proxy for aggregate economic activity and price dynamics that can affect default risk and required real returns.

• Sector ETF Prices and Returns: Includes a sector ETF identifier (sector_etf), the associated ETF price level (etf_price), and its monthly return (etf_return). These variables capture industry-level performance and sector-specific shocks that may not be fully reflected in firm-level data.

All macroeconomic and market-wide series are sampled or aggregated at a monthly frequency and merged with the bond panel by calendar month, ensuring that only information available at or before the end of month \(t\) is used to predict YTM changes between \(t\) and \(t+1\).

### 2.2 Feature Construction (Yutung)

We will standardize continuous features to have zero mean and unit variance. 

### 2.3 Data Splitting

In this project, we used a rolling window forecasting approach similar to our homework. At each year, we used a time-series split to create training, validation, and test sets. The training set consisted of 10 years of data, the validation set and test set each consisted of 1 year of data. When tuning is not involved, the validation set was combined with the training set to train the final model. 
For example, to predict YTM changes in 2012, we used data from 2002 to 2011 as the training set, data from 2012 as the validation set, and data from 2013 as the test set. Then, to predict YTM changes in 2013, we used data from 2003 to 2012 as the training set, data from 2013 as the validation set, and data from 2014 as the test set. This process was repeated until we predicted YTM changes in 2024.

### 2.4 Modeling Framework

#### 2.4.1 Regression Models

We implemented and compared the following machine learning models for regression task:

| Model | Description |
|--------|--------------|
|Linear Regression | Baseline linear model |
|ElasticNet | Combination of L1 and L2 | 
|Boosting (LightGBM) | Ensemble of weak learners |
|Multilayer Perceptron | Feedforward neural network |

Here, for the ElasticNet model, we tuned the hyperparameters alpha and l1_ratio using 10-fold cross-validation on the training set. However, we only tuned once based on the first training set and used the same hyperparameters for all other training sets to reduce computation time. This approach was applied to all tuning processes in this project. For the Boosting model, we tuned the hyperparameters using a grid search.

We also implemented a Stacked Regressor that combines predictions from Linear Regression, ElasticNet, and LightGBM models using linear regression as the meta-model to improve overall performance.

For the Multilayer Perceptron (MLP), we first standardized all continuous predictors using the training-set mean and standard deviation, and clipped extreme standardized values to a bounded range before feeding them into the network. The MLP is a small fully connected feedforward neural network with ReLU activation, trained with mean squared error loss and the Adam optimizer using early stopping based on validation loss. Due to the higher computational cost of neural networks, the MLP was estimated on a separate 80/20 time split rather than within the rolling-window framework, so its results are reported as a complementary single-split experiment rather than being directly comparable to the rolling-window averages of the other regression models.


#### 2.4.2 Classification Models

For classification task, we implemented and compared the following models:

| Model | Description |
|--------|--------------|
| Linear Regression        | Baseline linear model |
| ElasticNet               | Combination of L1 and L2 | 
| Boosting (LightGBM)      | Ensemble of weak learners |

Here, we also tuned hyperparameters for ElasticNet Classifier and Boosting Classifier using similar approaches as in regression task.

For the Stacked Classifier, we combined predictions from Logistic Regression, ElasticNet Classifier, and LightGBM Classifier using logistic regression as the meta-model.

### 2.5 Model Evaluation

We used a rolling window evaluation approach and calculated performance metrics on the test set for each year from 2012 to 2024. Then, the metrics were averaged to assess overall model performance.

For regression models, we evaluated performance using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Median Absolute Error (MedAE)
- $R^2$ 

For classification models, we evaluated performance using:

- Accuracy: Overall correctness of the model.
- Precision: Proportion of positive identifications that were actually correct.
- Recall: Proportion of actual positives that were identified correctly.


## 3. Results and Discussion

### 3.1 Regression Results

The table below summarizes the performance of different regression models on the test set:

| Model                  | MSE       | MAE      | MedAE    | $R^2$    |
|------------------------|-----------|----------|----------|---------|
| Elastic Net Regression | 0.000093  | 0.003167 | 0.002032 | 0.005956 |
| LGBM Regressor         | 0.000084  | 0.002817 | 0.001607 | 0.085135 |
| Linear Regression      | 0.000093  | 0.003171 | 0.002036 | 0.004982 |
| Stacked Regressor      | 0.000084  | 0.002841 | 0.001607 | 0.063947 |
| Multilayer Perceptron  | 0.000103  | 0.003265 | 0.001792 | 0.232484 |

The Elastic Net Regression model did not significantly outperform the Linear Regression model, indicating that regularization may not provide substantial benefits in this context. The LGBM Regressor achieved the lowest MSE and highest R^2, suggesting that ensemble methods can better capture complex relationships in the data. The Stacked Regressor also performed well, leveraging the strengths of multiple models but did not surpass the LGBM Regressor.
The multilayer perceptron (MLP) attains test errors between those of the linear models and LightGBM, providing additional evidence that some nonlinear structure is present in YTM changes, although tree-based ensembles appear to exploit it more effectively in our current setup.


### 3.2 Classification Results

As we mentioned before, we defined the target variable for classification as the direction of YTM changes (up, neutral, down) to mitigate the impact of outliers. The table below summarizes the performance of different classification models on the test set:

| Model                      | Accuracy | Precision | Recall   |
|---------------------------|----------|-----------|----------|
| Elastic Net Classification| 0.682760 | 0.694127  | 0.682760 |
| LGBM Classifier           | 0.713273 | 0.722421  | 0.713273 |
| Logistic Regression       | 0.682736 | 0.694145  | 0.682736 |
| Stacked Classifier        | 0.706753 | 0.718828  | 0.706753 |

In classification tasks, both the Elastic Net Classifier and Logistic Regression models exhibited similar performance, which is consistent with the regression results. The LGBM Classifier outperformed the other models, achieving the highest accuracy, precision, and recall. The Stacked Classifier also showed strong performance, although it did not exceed the LGBM Classifier.


### 4. Conclusion

In this project, we explored a range of machine learning models to predict corporate bond yield-to-maturity changes using firm-level fundamentals, market-based risk factors, and macroeconomic indicators. Our results show that ensemble methods such as LightGBM consistently outperform traditional linear models in both regression and classification tasks, while regularization techniques like ElasticNet do not provide material improvements over the linear regression benchmark in this setting. The multilayer perceptron achieves test errors between those of the linear models and LightGBM, suggesting that there is some nonlinear structure in YTM changes that a neural network can exploit, although tree-based ensembles appear to leverage this structure more effectively under our current feature set and evaluation scheme.

Overall, the performance gains across all models are modest: even the best-performing methods only reduce prediction errors by a limited margin relative to simple baselines. This indicates that a substantial portion of corporate bond YTM changes may be driven by noise or unobserved factors that are not captured by our variables. To achieve more meaningful improvements in predictive accuracy, future work may require cleaner and more granular data, richer feature representations (e.g., textual disclosures, market microstructure signals), or alternative modeling frameworks that more directly account for regime shifts and rare credit events.
