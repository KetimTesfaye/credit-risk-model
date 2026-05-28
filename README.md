Set-Content -Path README.md -Value @"
# Bati Bank Credit Scoring Engine

This repository contains an end-to-end credit risk modeling pipeline built for Bati Bank to support its upcoming Buy-Now-Pay-Later (BNPL) eCommerce partnership program.

## Credit Scoring Business Understanding

### 1. Basel II Accord Compliance: Interpretability and Documentation
The Basel II Accord framework establishes rigorous international standards governing risk measurement, asset classification, and mandatory documentation tracking. Under Basel II's Internal Ratings-Based (IRB) approach, banks are allowed to design their own risk estimation parameters, provided their internal predictive models are completely auditable. 

This regulatory pressure heavily enforces transparency:
* **Capital Requirements:** Model outputs directly calculate the minimum emergency capital reserves Bati Bank must maintain. A black-box framework cannot be legally validated by banking auditors, exposing the bank to heavy fines.
* **Audit Trails:** Every variable transformation (e.g., Weight of Evidence adjustments) must hold a clear lineage back to raw user transaction paths. 
* **The Right to an Explanation:** Credit officers must be able to cleanly dissect any risk probability score into clear consumer indicators (e.g., historical purchasing frequency drops) to explain credit limits or loan rejections.

### 2. RFM Proxy Target Variables & Inherent Strategic Risks
Because the partnering eCommerce platform provides raw transactional records instead of structured, mature loan repayment data, there is no historical "default" flag. To construct a machine learning classification engine, we must engineer a proxy target variable for default using Recency, Frequency, and Monetary (RFM) behavioral patterns. 

While necessary to establish a base risk label, this approach presents critical business risks:
* **The Mapping Misalignment:** A sudden collapse in a customer's purchasing recency or frequency may simply signify a shift to an alternative merchant or an app uninstallation—not a baseline financial insolvency or intent to default. This introduces a risk of **False Positives**, leading Bati Bank to deny loans to valid consumers, strangling market share growth.
* **Underestimating Delinquency Risks:** Conversely, high historical monetary volumes could reflect past behavior, masking current structural liquidity shocks. This creates **False Negatives**, leading to high loan default rates that erode underwriting profit margins.

### 3. Structural Modeling Architecture Trade-offs
Deploying an enterprise credit risk infrastructure requires evaluating traditional statistical scorecard frameworks against advanced tree-ensemble architectures:

| Evaluation Dimension | Traditional Framework: Logistic Regression + Weight of Evidence (WoE) | Advanced Machine Learning: Gradient Boosting (XGBoost / LightGBM) |
| :--- | :--- | :--- |
| **Mathematical Structure** | Linear and explicitly compartmentalized; continuous variables are manually binned into discrete intervals. | Non-linear tree ensemble; constructs thousands of sequential decision splits to define high-dimensional boundaries. |
| **Model Interpretability** | **Extremely High.** Coefficients translate directly to an operational credit scoring card easily vetted by credit staff. | **Low.** Deep tree interactions form a black-box environment by default. |
| **Regulatory Compliance** | **Seamless.** Directly aligns with traditional Basel II criteria out of the box; easily audited. | **Complex.** Requires post-hoc mathematical explainability layers (like SHAP values) to justify specific split factors. |
| **Handling Complex Patterns** | **Poor.** Fails to recognize non-linear behavior dynamics without manual engineering. | **Excellent.** Automatically extracts deep multi-variable cross-interactions from raw transaction fields. |
| **Predictive Performance** | **Moderate.** Risk of higher generalization errors if credit behavior profiles display complex shifts. | **Superior.** Minimizes operational loss by optimizing the separation between safe and high-risk applicants. |
"@