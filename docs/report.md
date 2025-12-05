
# Predicting Corporate Bond YTM Changes Using Machine Learning

Team members: Yuxi Geng, Yutung Lin, Qinqin Huang

## 1. Objective

The objective of this project is to predict corporate bond yield to maturity (YTM) changes using firm-level fundamentals, market-based risk factors, and macroeconomic indicators.  

By applying various machine learning regression models such as ElasticNet, Boosting and Multi-layer Perceptron, we aim to improve prediction accuracy compared to traditional linear regression models.

In addition, we used the change direction of YTM movements as a classification target since there are many outliers in the YTM change values that may affect regression performance. We applied classification models such as Logistic Regression, ElasticNet Classifier, and Boosting Classifier to predict the direction of YTM changes.


## 2. Methodology

### 2.1 Data Description

We utilize monthly data from Wharton Research Data Services (WRDS) from July 2002 to February 2025. The final panel contains approximately 854,769 bond–month observations for 12,117 unique bonds. We use about 50 predictors from four categories: bond-level characteristics, firm-level stock and accounting variables, macroeconomic indicators, and industry/sector variables (detailed in the appendix).

The regression target is the one-month change in corporate bond yield to maturity (YTM),
$$
\Delta \text{YTM}_i = \text{YTM}_{i,t} - \text{YTM}_{i,t-1},
$$
where $\text{YTM}_{i,t}$ is the yield to maturity of bond $i$ at the end of month $t$.

For classification, we use a three-way label indicating the direction of the YTM change:
$$
\text{YTM_Direction}_i =
\begin{cases}
\text{up}, & \text{if } \Delta \text{YTM}_i > 10^{-6}, \\
\text{neutral}, & \text{if } |\Delta \text{YTM}_i| \le 10^{-6}, \\
\text{down}, & \text{if } \Delta \text{YTM}_i < -10^{-6}.
\end{cases}
$$


### 2.2 Feature Construction

After preparing the bond-level, firm-level, sector-level, and macroeconomic datasets, we construct a monthly panel at the bond–firm–month level. This stage consists of dataset merging, temporal alignment with appropriate lags, missing-value handling, and feature normalization.

1. Merging and alignment  

We first merge CRSP stock data and Compustat accounting data at the firm-month level using PERMNO and month-end dates. Monthly macroeconomic variables (e.g., equity index, interest rates, volatility, GDP, CPI) are then merged by calendar month. Sector ETF prices and returns are linked using each issuer’s industry classification to proxy for sector-level conditions. Finally, bond-level data are merged using the issuer’s CUSIP-6 (issuer6) and month-end dates so that each bond-month observation is matched to the corresponding firm, sector, and macro information. Variables that are only observed at lower frequencies (e.g., quarterly accounting data) are lagged to avoid look-ahead bias.

2. Missing-value handling  

After merging, missing values are treated in a simple and consistent way. For categorical/dummy variables (credit rating buckets, rating-transition indicators, Compustat data-status flags), we impute missing entries using the cross-sectional monthly median, which coincides with the majority class for 0/1 dummies and preserves the monthly distribution across firms. For numeric variables, we also impute missing values using the cross-sectional median within each month; any remaining missing entries after this step are set to zero.

3. Normalization  

To make predictors comparable across firms and over time, we apply two types of transformations:

- **Rank-normalized features.**  
  Bond-level numeric predictors, stock-return and liquidity measures, accounting ratios, and growth indicators are rank-normalized cross-sectionally within each month. For a given variable \(x\), we compute
  \[
  \text{scaled\_rank} = 2 \times \frac{\text{rank}(x)}{N} - 1,
  \]
  which maps the cross-sectional ranks into the interval \([-1, 1]\). For binary indicators (e.g., upgrade/downgrade), we assign ranks so that 1 maps to 1 and 0 maps to –1.

- **Macroeconomic and yield-curve features.**  
  Macroeconomic variables and yield-curve variables (such as `sp500_ret`, `gdp_gr`, `cpi_infl`, `ir3m_chg`, `ir10y_chg`, `vix_chg`, `gs3m`, `term_spread`) are identical across firms in a given month, so cross-sectional rank normalization would collapse them into constants. Instead, these variables are standardized at the modeling stage using the mean and standard deviation computed from the training period only. This preserves meaningful time-series variation while avoiding data leakage from future observations.

### 2.3 Data Splitting

In this project, we used a rolling window forecasting approach. At each year, we used a time-series split to create training, validation, and test sets. The training set consisted of 10 years of data, the validation set and test set each consisted of 1 year of data. At each year, we tuned hyperparameters based on training set and validation set, then forecast target variables in the test set.

When tuning is not involved or cross-validation is adopted, the validation set was combined with the training set to train the final model. 
For example, to predict YTM changes in 2012, we used data from 2002 to 2011 as the training set, data from 2012 as the validation set, and data from 2013 as the test set. Then, to predict YTM changes in 2013, we used data from 2003 to 2012 as the training set, data from 2013 as the validation set, and data from 2014 as the test set. This process was repeated until we predicted YTM changes in 2024.

### 2.4 Modeling Framework

**2.4.1 Regression Models**

We implemented and compared the following machine learning models for regression task:

| Model | Description |
|--------|--------------|
|Linear Regression | Baseline linear model |
|ElasticNet | Combination of L1 and L2 | 
|Boosting (LightGBM) | Ensemble of weak learners |
|Multi-layer Perceptron | Feedforward neural network |

Here, for the ElasticNet model, we tuned the hyperparameters alpha and l1_ratio using 10-fold cross-validation on the training set. 

We also implemented a Stacked Regressor that combines predictions from Linear Regression, ElasticNet, and LightGBM models using linear regression as the meta-model to improve overall performance.

For the Multilayer Perceptron (MLP), we use the same rank-normalized predictors as in the other regression models (bond-level, firm-level, and sector-level features), together with the raw macro variables. The MLP is implemented as a small fully connected feed-forward network with ReLU activation functions and a single linear output node, trained with mean squared error loss and the Adam optimizer. 

Before the final evaluation, we conduct a grid search over the number of hidden layers, the number of units per layer, the learning rate, and the batch size using a 10-year training window and a 1-year validation window within each rolling block. The configuration that maximizes the average validation $R^2$ across all windows is selected as the tuned MLP. For the final rolling experiment, we fix this tuned architecture and, for each window, re-estimate the MLP on the combined 10-year training + 1-year validation period and evaluate it on the 1-year test period, using an internal hold-out split within the training window for early stopping. Thus, the MLP is directly comparable to the other regression models in terms of the rolling-window evaluation design, although it is considerably more computationally intensive and we keep the network architecture relatively small to control training time.

The MLP is not part of the stacked regressor due to its separate training procedure and high computational cost.

Notice that in each load, we normalized the macroeconomic variables based on the training set statistics. That's because these variables are identical across all firms in a given month, normalizing them using the entire dataset would lead to data leakage and normalizing them monthly would collapse them into constants, eliminating their informational content.

**2.4.2 Classification Models**

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

| Model              |      MSE     |    MAE     |   MedAE   |     $R^2$      |
|--------------------|--------------|------------|-----------|-------------|
| ElasticNet         | 0.000097     | 0.003107   | 0.001900  | -0.019272   |
| LGBMRegressor      | 0.000089     | 0.003069   | 0.001620  | -0.109083   |
| LinearRegression   | 0.000094     | 0.003226   | 0.002065  | 0.001753    |
| StackingRegressor  | 0.000086     | 0.002959   | 0.001591  | -0.008823   |
| Multilayer Perceptron  | 0.000105  | 0.003522 | 0.002168 | 0.064329 |

The Elastic Net Regression model does not significantly outperform the Linear Regression model, indicating that regularization alone does not provide substantial gains in this setting. The LGBM Regressor and the Stacking Regressor achieve slightly lower MSE than the pure Linear model, but their average out-of-sample $R^2$ remains close to zero (and sometimes slightly negative), suggesting only limited incremental predictability from these nonlinear ensembles.

The tuned multilayer perceptron (MLP) achieves the highest average out-of-sample $R^2 \approx 0.064$ across the rolling windows, indicating that the nonlinear network is able to capture some additional time-series variation in YTM changes. However, this comes at the cost of slightly higher MSE/MAE compared with the best tree-based and stacked models, and the overall magnitude of the $R^2$ remains modest. This suggests that, given our current feature set and network size, a feed-forward neural network provides only limited improvement in predictive accuracy over simpler linear and ensemble benchmarks.


### 3.2 Classification Results

As we mentioned before, we defined the target variable for classification as the direction of YTM changes (up, neutral, down) to mitigate the impact of outliers. The table below summarizes the performance of different classification models on the test set:


| Model                 | Accuracy  | Precision | Recall   |
|-----------------------|-----------|-----------|----------|
| Elastic Net Classifier | 0.687915 | 0.686969  | 0.687915 |
| LGBM Classifier        | 0.713470 | 0.719949  | 0.713470 |
| Logistic Regression    | 0.687858 | 0.687190  | 0.687858 |
| Stacking Classifier    | 0.713615 | 0.720544  | 0.713615 |

In classification tasks, both the Elastic Net Classifier and Logistic Regression models exhibited similar performance, which is consistent with the regression results. The LGBM Classifier outperformed the other models, achieving the highest accuracy, precision, and recall. The Stacked Classifier also showed strong performance, although it did not exceed the LGBM Classifier.


## 4. Conclusion

In this project, we explored various machine learning models to predict corporate bond yield to maturity changes using firm-level fundamentals, market-based risk factors, and macroeconomic indicators. Our findings indicate that ensemble methods like LightGBM provide noticeable gains in the classification task, where they outperform traditional linear models in terms of accuracy, precision, and recall. For the regression task, however, even flexible nonlinear models such as LightGBM, model stacking, and the tuned MLP deliver only modest improvements in MSE and relatively small positive out-of-sample $R^2$ compared with a simple linear benchmark.


## Appendix: 

### A.1 Implementation Details

1. Software and Libraries
- CPU: MacBook Pro (Apple M4 Pro)
- Programming Language: Python 3.13
- Libraries: pandas, numpy, scikit-learn, lightgbm, tensorflow/keras
- Code Repository: Github [https://github.com/QinqinAndMacaulayCat/ml_project/tree/main]

2. Computation Time For Each Model (Approximate)

| Model                      | Tuning                   | Time for Training and Prediction |
|----------------------------|--------------------------|----------------------------------|
| Linear Regression          |                          | 3.9 seconds                      |
| Elastic Net Regression     |                          | 1 minute 10 seconds              |
| LGBM Regressor             | 31 minutes 41.4 seconds  | 58.3 seconds                     |
| Stacked Regressor          |                          | 6 minutes 50 seconds             |
| Logistic Regression        |                          | 1 minute 30 seconds              |
| Elastic Net Classification | 83 minutes 59.8 seconds  | 23 minutes 40.7 seconds          |
| Boosting Classifier        | 114 minutes 48.8 seconds | 1 minute 27.9 seconds            |
| Stacked Classifier         |                          | 107 minutes 11.9 seconds         |
| Multilayer Perceptron      | 38 minutes 48 seconds    | 3 minutes 16 seconds             |

### B.1 Predictor List

1. Key Columns

| Field Name | Full Name | Description |
|------------|-----------|-------------|
| CUSIP | Committee on Uniform Securities Identification Procedures | Unique 9-digit identifier for each bond, used to track bond-level attributes and returns. |
| PERMNO | Permanent Number | Unique firm-level identifier from CRSP, linking each bond to its issuing firm. |
| Date | End of Month Date | The month-end observation date for which bond yields, returns, and explanatory variables are recorded. |

2. Bond-level features

- Time to Maturity (tmt): the remaining time to maturity of the bond in years.
- Coupon Rate (coupon): the annual coupon rate of the bond.
- t_spread: the weighted average bid-ask spread of the bond which measures liquidity.
- return_eom: bond return of last month. 
- Credit Ratings: One-hot encoded variables for ratings from AAA to D (rating_A, rating_AA, ..., rating_D). The ratings are weighted averages ratings from WRDS Bond Returns database rather than S&P, Moody's, or Fitch alone.
- Rating Change Indicators: Binary variables indicating whether there was an upgrade or downgrade in the bond's credit rating in the past month (upgrade, downgrade).


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


We incorporate firm-level information from both daily stock data (CRSP) and quarterly accounting data (Compustat). These datasets provide market-based signals and fundamental characteristics of the issuing firms. We then construct additional financial ratios and growth indicators used as predictors.

(1) Stock Market Features (Monthly Aggregated)

- ret — Monthly stock return (accumulated from daily returns).
- vol — Monthly return volatility (std of daily log returns).
- dvol — Dollar trading volume (price × shares traded).
- turnover — Trading turnover (shares traded ÷ shares outstanding).
- bidask — Bid–ask spread based on daily midquotes.
- numtrades — Number of trades in the month.
- mktcap — Month-end market capitalization.
- price_mean — Average daily stock price.

These features reflect firm-level market performance, trading activity, and liquidity conditions.

(2) Accounting-Based Derived Indicators (Monthly Filled)

- log_atq — Log total assets (firm size proxy).
- lev_total — Leverage ratio = total liabilities ÷ total assets.
- equity_ratio — Equity-to-assets ratio.
- roa — Profitability = net income ÷ total assets.
- profit_margin — Net profit margin = net income ÷ revenue.
- int_coverage — Ability to service debt = operating income ÷ interest expense.
- mkt_cap — Accounting-based market cap = prccq × cshoq.
- market_to_book (m2b) — Valuation ratio = (market cap ÷ common equity).
These variables summarize capital structure, profitability, solvency, and valuation.


(3) Growth Indicators (QoQ)

- atq_growth — Asset growth (decomposition of balance-sheet expansion).
- revtq_growth — Revenue growth (sales momentum).
- niq_growth — Net income growth (profitability momentum).
These features capture firm momentum and fundamental acceleration, which are known predictive signals in credit and bond return literature.

| Variable          | Mean      | Std       | Min        | 25%       | 50%       | 75%       | Max        |
|------------------|-----------|-----------|------------|-----------|-----------|-----------|------------|
| stock_ret        | 0.00081  | 0.1162    | -0.3446    | -0.052  | 0.00206  | 0.05164   | 0.3881     |
| stock_vol        | 0.02420   | 0.01729   | 0          | 0.01106   | 0.01978   | 0.03373   | 0.07639    |
| dvol             | 4.720e8   | 1.338e9   | 0          | 3.564e6   | 2.814e7    | 2.269e8   | 9.083e9    |
| price_mean | 27.89     | 33.39     | 0.4247     | 7.503     | 17.41     | 34.67     | 202.7      |
| turnover         | 10.55     | 20.04     | 0.1046     | 2.171     | 5.032     | 10.39     | 151.0      |
| bidask           | 0.03700   | 0.03154   | 0.00136   | 0.01481   | 0.02762   | 0.04922   | 0.1571     |
| numtrades        | 2.353e4   | 7.754e4   | 0          | 0         | 0         | 5060      | 5.375e5    |
| stock_mktcap     | 2.558e6   | 7.306e6   | 2612       | 6.575e4   | 2.860e5    | 1.336e6   | 5.040e7    |
| log_atq          | 5.966     | 2.271     | 1.059      | 4.265     | 5.935     | 7.575     | 11.88      |
| lev_total        | 0.5453    | 0.2667    | 0.04291    | 0.3332    | 0.5433    | 0.7454    | 1.437      |
| equity_ratio     | 0.4682    | 0.2518    | 0.03758    | 0.2655    | 0.4584    | 0.6722    | 0.9650     |
| roa              | -0.011  | 0.06202   | -0.4176    | -0.001 | 0.00419  | 0.01634   | 0.1046     |
| profit_margin    | -0.067  | 0.5065    | -4.911     | -0.032  | 0.03624   | 0.09573   | 0.8525     |
| int_coverage     | 7.883     | 10.21     | -9.713     | 2.324     | 4.903     | 9.881     | 78.47      |
| mkt_cap          | 3631      | 2.366e4   | 0.0150   | 55.60     | 249.9     | 1257      | 4.330e6    |
| m2b   | 2.865     | 3.123     | 0.2618     | 1.144     | 1.846     | 3.275     | 26.64      |
| atq_growth       | 0.0306   | 0.2174    | -1.000     | -0.023  | 0.00585  | 0.04315   | 4.966      |
| revtq_growth     | 0.0594   | 0.3745    | -1.000     | -0.058  | 0.01596   | 0.1119    | 5.000      |
| niq_growth       | -0.111   | 1.158     | -5.000     | -0.538   | -0.033  | 0.2378    | 5.000      |



4. Macroeconomic features 
- Equity Market Return (sp500):  
  Monthly level of a broad U.S. equity index (S&P 500), used to capture overall stock market conditions and risk appetite.

- Short-Term Interest Rate (ir3m):  
  Monthly average of the 3-month Treasury yield, serving as the short end of the risk-free term structure and a proxy for monetary policy stance.

- Long-Term Interest Rate (ir10y):  
  Monthly average of the 10-year Treasury yield, capturing long-term discount rates relevant for bond valuation.

- Implied Volatility (vix):  
  Monthly average of the VIX index, measuring market-implied equity volatility and shifts in aggregate risk aversion.

- Real Economic Activity (gdp):  
  Real GDP level (or growth, depending on transformation) aligned to the month of observation, summarizing aggregate economic conditions.

- Inflation (cpi):  
  Consumer Price Index level (or inflation rate) matched to each month, capturing the inflation environment relevant for nominal yields and real required returns.

| Variable | Count |   Mean   |   Std   |    Min   |    25%    |    50%    |    75%    |    Max    |
|:---------|------:|---------:|--------:|---------:|----------:|----------:|----------:|----------:|
| SP500    |   335 | 2185.71  | 1418.34 |  735.09  | 1191.41   | 1454.60   | 2772.33   | 6840.20   |
| IR3M     |   335 |    2.07  |    2.00 |    0.00  |    0.12   |    1.51   |    4.21   |    6.15   |
| IR10Y    |   335 |    3.44  |    1.37 |    0.54  |    2.30   |    3.53   |    4.49   |    6.67   |
| VIX      |   335 |   20.29  |    7.82 |    9.51  |   14.45   |   18.43   |   24.44   |   59.89   |
| GDP      |   328 | 17764.90 | 2911.28 | 12703.74 | 15670.88  | 17035.11  | 20070.68  | 23770.98  |
| CPI      |   333 |  227.69  |   42.89 |  162.00  |  191.70   |  227.17   |  252.56   |  324.37   |

5. Industry features

- Sector ETF Price and Return (price, return):  
  For each bond, we link a sector ETF based on its GICS sector, and use the ETF’s monthly price (`price`) and monthly return (`return`) to proxy for sector-level performance and sector-specific shocks that may not be fully captured by firm-level variables.

| Variable | Count |  Mean   |   Std   |    Min    |    25%    |    50%    |    75%    |    Max    |
|:---------|------:|--------:|--------:|----------:|----------:|----------:|----------:|----------:|
| Price    |  3128 |  44.03  |  39.40  |   4.57    |  18.23    |  29.43    |  54.97    | 300.68    |
| Return   |  3117 |   0.008 |   0.055 |  -0.3437  | -0.0217   |  0.0110   |  0.0390   |  0.3076   |


### B.2 Classification Confusion Matrix

The confusion matrixs are as follows:

1. Logistic Classification

| Actual \ Pred |    Up   |   Down  | Neutral |
|---------------|---------|---------|---------|
| Up            | 155810  | 103344  |   200   |
| Down          |  52584  | 207780  |   286   |
| Neutral       |   4410  |   6969  |    79   |


2. Elastic Net Classification

| Actual \ Pred |    Up   |   Down  | Neutral |
|---------------|---------|---------|---------|
| Up            | 155781  | 103385  |   188   |
| Down          |  52524  | 207861  |   265   |
| Neutral       |   4418  |   6975  |    65   |


3. LGBM Classifier

| Actual \ Pred |    Up   |   Down  | Neutral |
|---------------|---------|---------|---------|
| Up            | 176082  |  82742  |   530   |
| Down          |  62480  | 197594  |   576   |
| Neutral       |   2382  |   3005  |  6071   |


4. Stacked Classifier

| Actual \ Pred |    Up   |   Down  | Neutral |
|---------------|---------|---------|---------|
| Up            | 178684  |  80475  |   195   |
| Down          |  64062  | 196423  |   165   |
| Neutral       |   3217  |   4005  |  4236   |



