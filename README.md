# Netflix Customer Churn: Drivers and Prediction

Which customer behaviours are linked to cancelling a streaming subscription, and how well can a model predict who will churn? This notebook runs hypothesis tests on 8 variables, compares 12 classification models, and estimates customer lifetime value and retention campaign ROI by subscription tier.

This was a school project. The dataset is synthetic, so the model scores below show that the pipeline works, not how well churn can be predicted at a real company.

## Data

[Netflix Customer Churn dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) (Kaggle, Abdul Wadood): 5,000 customers, 14 columns, no missing values, 50.3% churned. A copy is in `data/` so the notebook runs as downloaded. No licence is stated on the Kaggle page.

## Findings

**What is linked to churn** (tests at alpha = 0.05):

| Variable | Result |
|---|---|
| Watch hours | r = -0.48, customers who watch less churn more |
| Days since last login | r = +0.47, longer gaps mean more churn |
| Average watch time per day | r = -0.27 |
| Subscription type | Basic 61.8% churned, Standard 45.4%, Premium 43.7% (Cramer's V = 0.16) |
| Payment method | Crypto 59.7% and gift card 57.8% churned, credit and debit card about 44% (Cramer's V = 0.14) |
| Age, gender, region | No significant link (p > 0.7) |

Engagement is the main signal. Plan and payment method matter less, and demographics do not matter in this data.

**Models** (80/20 split, 1,000 test customers):

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic regression | 0.901 | 0.972 |
| SVM | 0.920 | 0.981 |
| Decision tree (best of 5 settings) | 0.992 | 0.993 |
| Random forest (best of 3 settings) | 0.985 | 0.998 |
| Gradient boosting (best of 2 settings) | 0.992 | 0.999 |

**Business estimate** (from stated assumptions, see below): Premium customers have about 3 times the lifetime value of Basic, and Basic customers are worth less than their acquisition cost. A retention campaign aimed at all high-risk customers loses money at a 15% success rate (ROI -47%) and breaks even at about 28%.

## Things to know before reading the scores

- **The scores are not realistic.** Churn here follows a near-fixed pattern in the data. A depth-3 decision tree on just two columns (watch hours, days since last login) already reaches about 90% accuracy in 5-fold cross-validation, and the best model's predicted probabilities are almost all near 0 or near 1. Real churn data is far noisier, and 0.99 accuracy should not be expected from it.
- **The best model was picked using the test set.** The cross-validation on the training set (ROC-AUC 0.999) agrees with the test score, but a cleaner setup would choose the model on validation data only.
- **The lifetime value figures come from assumptions, not from the data.** The notebook assumes a 60% gross margin, $120 acquisition cost, $25 cost per targeted customer and a 15% campaign success rate. It also treats each tier's churn share as an annual rate to get a lifespan, which is a modelling choice. Change those inputs and the dollar figures change.
- The high-risk counts and revenue at risk are for the 1,000 test customers only, not all 5,000.

## Run it

    pip install -r requirements.txt
    cd notebooks
    jupyter notebook 01_netflix_churn_analysis.ipynb

Run from the `notebooks/` folder so the relative data path works. Results are saved to `outputs/`. Exact scores can differ in the third decimal between scikit-learn versions.

## Files

- `notebooks/01_netflix_churn_analysis.ipynb`: the full analysis with outputs
- `data/netflix_customer_churn.csv`: the dataset
- `outputs/model_comparison_results.csv`: scores for all 12 models
- `outputs/statistical_analysis_results.json`: hypothesis test results
