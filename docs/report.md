
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


### 2.2 Feature Construction (Yutung)
To enhance bond yield prediction, we construct a comprehensive set of features from firm-level, market-level, and macroeconomic information. All features are aligned to month-end and lagged by one month to prevent look-ahead bias, reflecting the information available to investors at the time of prediction.

1. Firm-Level Stock Market Features (Monthly Aggregated)
    Derived from CRSP daily data and aggregated to monthly frequency:

    (a) Price & Return
        •	stock_ret — Monthly stock return (product of daily returns − 1).
        •	stock_vol — Monthly volatility (standard deviation of daily log returns).
        •	dvol — Total monthly dollar trading volume.
        •	stock_price_mean — Average daily price within the month.

    (b) Liquidity Measures
        •	turnover — Monthly average turnover.
        •	bidask — Monthly average bid–ask spread.
        •	numtrades — Number of trades summed over the month.
        •	stock_mktcap — Market capitalization at month end (PRC × SHR_OUT).

2. Quarterly Accounting Fundamentals (Forward-Filled Monthly)
    Original Compustat variables resampled to month-end:

    (a) Raw Accounting Fundamentals
        •	atq — Total assets
        •	ltq — Total liabilities
        •	ceqq — Common equity
        •	cshoq — Shares outstanding
        •	niq — Net income
        •	oibdpq — Operating income before depreciation
        •	revtq — Revenue
        •	xintq — Interest expense
        •	prccq — Closing price (quarterly)

    (b) Derived Financial Ratios
        •	log_atq — Log total assets
        •	lev_total — Leverage ratio (ltq / atq)
        •	equity_ratio — Equity-to-assets ratio (ceqq / atq)
        •	roa — Return on assets (niq / atq)
        •	profit_margin — Net margin (niq / revtq)
        •	int_coverage — Interest coverage (oibdpq / xintq)
        •	accounting_mktcap — Accounting-based market cap (prccq × cshoq)

    (c) Growth Rates (QoQ)
        •	atq_growth — Asset growth
        •	revtq_growth — Revenue growth
        •	niq_growth — Net income growth

3. Sector-Level ETF Features
    Monthly performance of sector ETFs (mapped via Global Industry Classification):
        •	etf_price — Sector ETF price at month-end
        •	etf_return — Monthly ETF return


4. Macroeconomic Features (Monthly)
    We transform macro variables into economically meaningful derivatives:

    (a) Growth Rates
        •	sp500_ret — Monthly S&P 500 return
        •	gdp_gr — GDP growth rate
        •	cpi_infl — Inflation rate (CPI growth)

    (b) Interest Rate Dynamics
        •	ir3m_chg — Change in short-term interest rate (3-month Treasury)
        •	ir10y_chg — Change in long-term yield (10-year Treasury yield)

    (c) Market Stress Indicator
        •	vix_chg — Monthly change in VIX

5. Bond-Specific Features

    Derived directly from bond dataset:
        •	tmt — Time to maturity
        •	coupon — Coupon rate
        •	t_spread — Bond’s credit spread relative to its benchmark
        •	ytm — Yield-to-maturity (target variable)
        •	bond_ret — End-of-month bond return
        •	rating_AA, rating_A, … rating_D — Credit rating dummies
        •	upgrade, downgrade — Rating transition indicators

6. Missing-Value Handling & Normalization
    We treat variables as either categorical or numerical:

    Categorical features
        •	costat_bin
        •	Bond rating dummies: rating_A, rating_B, …
        •	Rating transition dummies: upgrade, downgrade

    Binary categoricals → fill per-month median, then encode as {–1, +1}

    Numeric features
    We first fill NaN with median values within each month, then divide numeric variables into two groups depending on whether they should be rank-normalized or kept in raw form

    | Type  | Features                                           | Normalization                               |
    |-------|------------------------------------------------------------|----------------------------------------------|
    | Rank-normalized | stock features, fundamentals, bond features...     | Cross-sectional rank scaled to [−1, 1] |
    | Not normalized  | macro features                                      | Kept in raw form                        |


    After median filling, remaining NaNs are filled with 0.
    This process ensures consistent feature availability while preventing information leakage.

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
|Multi-layer Perceptron | Feedforward neural network |

Here, for the ElasticNet model, we tuned the hyperparameters alpha and l1_ratio using 10-fold cross-validation on the training set. However, we only tuned once based on the first training set and used the same hyperparameters for all other training sets to reduce computation time. This approach was applied to all tuning processes in this project. For the Boosting model, we tuned the hyperparameters using a grid search.

We also implemented a Stacked Regressor that combines predictions from Linear Regression, ElasticNet, and LightGBM models using linear regression as the meta-model to improve overall performance.


#### 2.4.2 Classification Models

For classification task, we implemented and compared the following models:

| Model | Description |
|--------|--------------|
|Logistic Regression | Linear classification model |
|ElasticNet Classifier | Combination of L1 and L2 |
|Boosting Classifier (LightGBM) | Ensemble of weak learners |

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

| Model                 | MSE       | MAE      | MedAE    | $R^2$        |
|-----------------------|-----------|----------|----------|-----------|
| Elastic Net Regression | 0.000093 | 0.003167 | 0.002032 | 0.005956  |
| LGBM Regressor        | 0.000084 | 0.002817 | 0.001607 | 0.085135  |
| Linear Regression     | 0.000093 | 0.003171 | 0.002036 | 0.004982  |
| Stacked Regressor     | 0.000084 | 0.002841 | 0.001607 | 0.063947  |

The Elastic Net Regression model did not significantly outperform the Linear Regression model, indicating that regularization may not provide substantial benefits in this context. The LGBM Regressor achieved the lowest MSE and highest R^2, suggesting that ensemble methods can better capture complex relationships in the data. The Stacked Regressor also performed well, leveraging the strengths of multiple models but did not surpass the LGBM Regressor.

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

In this project, we explored various machine learning models to predict corporate bond yield to maturity changes using firm-level fundamentals, market-based risk factors, and macroeconomic indicators. Our findings indicate that ensemble methods like LightGBM outperform traditional linear models in both regression and classification tasks. Regularization techniques such as ElasticNet did not yield significant improvements over linear models in this context. However, the improvement was modest, suggesting that further enhancements such as more accurate data, additional features, or alternative modeling approaches may be necessary to achieve substantial gains in predictive accuracy. 
