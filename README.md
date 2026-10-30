Credit Default Prediction in Peer-to-Peer Lending

Using Textual Loan Descriptions to Enhance Credit Default Prediction in Peer-to-Peer Lending using Machine Learning Methods

MSc Data Science dissertation, University of Salford, 2025-2026. Completed with Distinction. Supervisor: Prof. Usman Isiyaku.

Do borrowers' written loan descriptions help predict whether a loan will default, beyond what the standard application data already tells us?

Contents:-
Overview
Research questions
Data
Method
Results
Key findings
Limitations and future work
Tools
How to run
Author
Overview

Peer-to-peer lending platforms such as Lending Club use credit risk models to decide which applications to approve. Those models usually rely on structured variables like income, debt-to-income ratio and FICO score. Many applicants also write a short title and free-text description explaining why they want the loan, and that text is normally ignored.

This project compares models built on three feature sets: structured data only, text only, and both combined. It measures how much the text adds, which words and phrases are linked to default, and whether the gain is large enough to matter in practice.

Research questions
RQ1: How much does adding loan titles and descriptions to structured variables improve default prediction?
RQ2: How do structured-only, text-only and combined models compare on ROC-AUC, PR-AUC and F1-score?
RQ3: Which text characteristics (having a description, its length, common terms) are most associated with default, and do they add information beyond structured predictors?
RQ4: Are the gains large enough to justify the extra processing and governance effort in a real credit pipeline?
Data
Curated Lending Club granting dataset with 1,347,681 loan applications and 15 variables
Only loans with a resolved outcome (fully paid, or defaulted/charged off) are used
Only information available at application time is used as predictors. Payment behaviour, internal grades and interest rates are excluded to avoid target leakage.
Binary target: default vs fully paid, with a default rate of about 19.98%
Only 8.8% of loans have a non-empty description
The dataset is not included here because of its size. [Add the link to where it can be downloaded.]

Structured features: income, debt-to-income ratio, loan amount, FICO score, credit experience indicator, employment length, loan purpose, home ownership and state. Text features: loan title and description.

Method

The project follows the CRISP-DM process, from problem definition and exploratory analysis through to evaluation.

Feature engineering: title and description are joined into one text field and converted with TF-IDF (up to 5,000 unigrams and bigrams). Three simple indicators are added: whether a description exists, its character length and its word count.
Pipelines: scikit-learn ColumnTransformer and Pipeline objects apply scaling, encoding and TF-IDF inside each model, so no information leaks from the test data.
Class imbalance: handled with class weights (balanced for logistic regression, balanced_subsample for random forest) instead of resampling.
Split: stratified 80/20 train-test split, giving 1,078,144 training and 269,537 test loans with the same default rate in each.
Models: logistic regression and random forest on structured features, a text-only logistic regression, and combined structured and text logistic regression models. Random forest was used as a structured-only comparison.
Tuning and validation: the combined model was tuned with RandomizedSearchCV using PR-AUC as the objective, inside a nested 3-fold cross-validation. The inner loop picks hyperparameters and the outer loop measures performance on data the tuning never saw.
Metrics: ROC-AUC, PR-AUC, F1-score and confusion matrices.
Threshold analysis: an F1-optimal threshold and a cost-based threshold, using an illustrative 5:1 cost for a missed default versus a wrongly declined loan.
Results

All figures are on the held-out test set.

Model	ROC-AUC	PR-AUC	F1 (threshold 0.5)
Logistic regression, structured only	0.6618	0.3162	0.3904
Random forest, structured only	0.6617	0.3163	0.3911
Logistic regression, text only	0.5486	0.2206	0.3337
Logistic regression, structured + text	0.6630	0.3156	0.3915
Tuned logistic regression, structured + text	0.6641	0.3176	0.3922

Nested cross-validation for the tuned model gave a mean PR-AUC of about 0.317 with low variance, which is consistent with the test result.

Thresholds for the tuned model: the F1-optimal threshold was about 0.509 and the cost-optimal threshold about 0.446 under the illustrative cost ratio.

Key findings
Structured data does most of the work. Logistic regression and random forest perform almost identically on it.
Text alone is weak but above chance (ROC-AUC 0.549).
Adding text helps a little. The tuned combined model beats the structured baseline by roughly 0.002 to 0.003 ROC-AUC, a small but consistent gain.
Risk drivers from the structured model: higher debt-to-income ratio, small-business purpose, renting and short or missing employment history raise default risk, while higher FICO score, home improvement or wedding purposes and longer employment lower it.
Risk signals in the text: phrases about financial distress ("medical bills", "late payments", "lost my job"), small-business ventures and credit card or collection debt are linked to higher default.
Why the gain is small: over 90% of loans have no description, those that exist are short, and TF-IDF captures word patterns but not deeper meaning.
Limitations and future work

Limitations

Text is missing for most loans, which caps what any text model can add.
Only logistic regression and random forest were used, and random forest was not tuned as extensively or applied to the combined feature set.
The data comes from one platform and one period, so results may not carry over to other lenders or economic conditions.
The cost values are illustrative, not based on real money amounts.

Future work

Test richer text representations such as pre-trained transformer language models
Compare gradient boosting and calibrated probability models
Add time-based and macroeconomic features and test on other platforms
Link model scores to a simulated portfolio to estimate expected losses
Check fairness across groups, since text can encode sensitive information

Ethics note: using free-text personal information in credit decisions raises governance, transparency and fairness questions, which the dissertation discusses alongside the results.

Tools

Python, Pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, TfidfVectorizer, LogisticRegression, RandomForestClassifier, RandomizedSearchCV), Matplotlib


How to run
bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
# download the dataset into the data/ folder, then open the notebook
jupyter notebook
Author

Pratham Sawant LinkedIn | sawantpratham2102@gmail.com

The dissertation PDF is available on request.
