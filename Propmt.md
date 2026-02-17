I am participating in a data science hackathon called Innofusion'26.
I need you to build Section 4 (Machine Learning) of a Google Colab
notebook. Here are the complete details:

=====================================================
DATASET INFORMATION
=====================================================

The CSV file is named: customer_behavior.csv
It has the following columns:

- Customer_ID        : string (drop this)
- Age                : integer
- Gender             : categorical (Male/Female/Non-binary/etc.)
- Income_Level       : categorical (High/Middle)
- Marital_Status     : categorical (Single/Married/Divorced/Widowed)
- Education_Level    : categorical (High School/Bachelor's/Master's)
- Occupation         : categorical (High/Middle)
- Location           : string (drop this)
- Purchase_Category  : categorical (Electronics/Clothing/etc.)
- Purchase_Amount    : string with $ sign → needs cleaning to float
- Frequency_of_Purchase : integer
- Purchase_Channel   : categorical (Online/In-Store/Mixed)
- Brand_Loyalty      : integer (1-5)
- Product_Rating     : integer (1-5)
- Time_Spent_on_Product_Research(hours) : float
- Social_Media_Influence : categorical (None/Low/Medium/High)
- Discount_Sensitivity   : categorical
                          (Not Sensitive/Somewhat Sensitive/Very Sensitive)
- Return_Rate        : integer (0-2)
- Customer_Satisfaction  : integer (1-10)
- Engagement_with_Ads    : categorical (None/Low/Medium/High)
- Device_Used_for_Shopping : categorical (Smartphone/Tablet/Desktop)
- Payment_Method     : categorical (Credit Card/Debit Card/PayPal/etc.)
- Time_of_Purchase   : date string (drop this)
- Discount_Used      : boolean (TRUE/FALSE)
- Customer_Loyalty_Program_Member : boolean (TRUE/FALSE)
- Purchase_Intent    : categorical (TARGET VARIABLE)
                      (Need-based/Impulsive/Planned/Wants-based)
- Shipping_Preference : categorical (Standard/Express/No Preference)
- Time_to_Decision   : integer

=====================================================
TARGET VARIABLE
=====================================================

Target column → Purchase_Intent
Classes:
  - Need-based
  - Impulsive
  - Planned
  - Wants-based

This is a MULTICLASS CLASSIFICATION problem.

=====================================================
WHAT TO BUILD - STEP BY STEP
=====================================================

Build a clean, well-commented Python script / Colab notebook
with the following sections:

---

SECTION 4.1 — LOAD & PREPARE DATA
- Load customer_behavior.csv using pandas
- Clean Purchase_Amount: remove "$" and spaces, convert to float
- Drop columns: Customer_ID, Location, Time_of_Purchase
- Convert Discount_Used and Customer_Loyalty_Program_Member
  from TRUE/FALSE strings to 1/0 integers

---

SECTION 4.2 — ENCODE CATEGORICAL VARIABLES
- Use LabelEncoder from sklearn to encode these columns:
    Gender, Income_Level, Marital_Status, Education_Level,
    Occupation, Purchase_Category, Purchase_Channel,
    Social_Media_Influence, Discount_Sensitivity,
    Engagement_with_Ads, Device_Used_for_Shopping,
    Payment_Method, Shipping_Preference
- Also encode the TARGET column Purchase_Intent with LabelEncoder
  but save the encoder separately as "label_encoder_target"
  so we can inverse_transform later
- Print the class mapping like:
    0 = Impulsive
    1 = Need-based
    2 = Planned
    3 = Wants-based

---

SECTION 4.3 — FEATURE SELECTION
- Define features X using ONLY these 11 columns:
    Age, Purchase_Amount, Frequency_of_Purchase,
    Customer_Satisfaction, Brand_Loyalty,
    Discount_Sensitivity, Income_Level,
    Engagement_with_Ads, Social_Media_Influence,
    Product_Rating, Time_Spent_on_Product_Research(hours)
- Define target y = Purchase_Intent (encoded)
- Print shape of X and y

---

SECTION 4.4 — TRAIN TEST SPLIT
- Split X and y:
    test_size = 0.20
    random_state = 42
    stratify = y  (important for class balance)
- Print train size and test size

---

SECTION 4.5 — TRAIN 3 MODELS

Model 1: Logistic Regression
- Use LogisticRegression(max_iter=1000, random_state=42)
- Fit on training data
- Predict on test data
- Store accuracy as lr_accuracy

Model 2: Random Forest
- Use RandomForestClassifier(
      n_estimators=100,
      max_depth=10,
      random_state=42,
      class_weight='balanced'
  )
- Fit on training data
- Predict on test data
- Store accuracy as rf_accuracy

Model 3: XGBoost
- Install xgboost if needed: !pip install xgboost -q
- from xgboost import XGBClassifier
- Use XGBClassifier(
      n_estimators=200,
      max_depth=4,
      learning_rate=0.1,
      subsample=0.8,
      colsample_bytree=0.8,
      use_label_encoder=False,
      eval_metric='mlogloss',
      random_state=42
  )
- Fit on training data
- Predict on test data
- Store accuracy as xgb_accuracy

---

SECTION 4.6 — MODEL COMPARISON CHART
- Create a horizontal bar chart comparing all 3 models
- X axis = Accuracy (0 to 1)
- Y axis = Model names
- Use colors: ['#4C72B0', '#55A868', '#C44E52']
- Add accuracy value labels at end of each bar
- Title: "Model Accuracy Comparison"
- Add a vertical dashed line at the highest accuracy
- Save chart as "model_comparison.png"

---

SECTION 4.7 — BEST MODEL EVALUATION
- Identify best model automatically by comparing
  lr_accuracy, rf_accuracy, xgb_accuracy
- Print which model won and its accuracy

- Plot Confusion Matrix for the best model:
    - Use seaborn heatmap
    - Use inverse_transform on labels to show
      actual class names (Impulsive/Need-based/etc.)
    - Color map: 'Blues'
    - Title: "Confusion Matrix - [Best Model Name]"
    - Save as "confusion_matrix.png"

- Print Classification Report showing:
    precision, recall, f1-score for each class
    Also print overall accuracy, macro avg, weighted avg

---

SECTION 4.8 — FEATURE IMPORTANCE CHART
- Get feature importances from the Random Forest model
  (use rf model always for this even if xgb wins,
   because RF feature importance is more interpretable)
- Create a horizontal bar chart:
    - Sort by importance descending
    - Use color: '#4C72B0'
    - Title: "Top Features Influencing Purchase Intent"
    - X axis label: "Feature Importance Score"
    - Y axis label: "Features"
    - Add value labels at end of each bar
    - Save as "feature_importance.png"

---

SECTION 4.9 — CROSS VALIDATION
- Run 5-fold cross validation on the BEST model
- Print mean accuracy and standard deviation like:
  "Cross-Validation Accuracy: 0.XX ± 0.XX"

---

SECTION 4.10 — BUSINESS SUMMARY
- Print a clean formatted summary box like:

  ╔══════════════════════════════════════════════════╗
  ║          ML MODEL SUMMARY                        ║
  ╠══════════════════════════════════════════════════╣
  ║ Target Variable  : Purchase Intent               ║
  ║ Best Model       : [model name]                  ║
  ║ Test Accuracy    : XX.XX%                        ║
  ║ Cross-Val Score  : XX.XX% ± XX.XX%               ║
  ║ Top Feature      : [feature name]                ║
  ║ 2nd Top Feature  : [feature name]                ║
  ║ 3rd Top Feature  : [feature name]                ║
  ╠══════════════════════════════════════════════════╣
  ║ BUSINESS INSIGHTS:                               ║
  ║ 1. [Auto-generated based on top feature]         ║
  ║ 2. [Auto-generated based on 2nd feature]         ║
  ║ 3. [Auto-generated based on model accuracy]      ║
  ╚══════════════════════════════════════════════════╝

=====================================================
CODE REQUIREMENTS
=====================================================

1. Every section must have a clear markdown-style
   comment header like:
   # ── SECTION 4.1: LOAD & PREPARE DATA ──────────

2. Every important step must have an inline comment
   explaining what it does

3. After each section print a success message like:
   print("✅ Section 4.1 complete")

4. All charts must:
   - Have titles
   - Have axis labels
   - Be saved as .png files
   - Use plt.tight_layout() before saving
   - Use figsize=(10, 6) minimum

5. Use random_state=42 everywhere for reproducibility

6. All code must run on Google Colab without errors

7. At the very top of the file add:
   !pip install xgboost -q

8. Import all libraries at the top:
   pandas, numpy, matplotlib, seaborn,
   sklearn (LabelEncoder, train_test_split,
   LogisticRegression, RandomForestClassifier,
   classification_report, confusion_matrix,
   cross_val_score, accuracy_score),
   xgboost (XGBClassifier)

=====================================================
OUTPUT FILES EXPECTED
=====================================================

1. model_comparison.png
2. confusion_matrix.png
3. feature_importance.png
4. Printed classification report in console
5. Printed business summary box in console

=====================================================
CODING STYLE
=====================================================

- Clean and readable
- No unnecessary complexity
- Beginner friendly comments
- No functions or classes needed,
  just sequential clean code blocks
- Each section should work independently
  if run as a separate cell in Colab

Now generate the complete Python code following
all instructions above exactly.