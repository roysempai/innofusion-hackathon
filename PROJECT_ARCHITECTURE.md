# 🏆 Innofusion'26 Hackathon - ML Project Architecture

## 📋 Project Overview

**Project Name**: Customer Purchase Intent Classification  
**Competition**: Innofusion'26 Data Science Hackathon  
**Task**: Section 4 - Machine Learning Pipeline  
**Problem Type**: Multiclass Classification (4 classes)  
**Target Variable**: Purchase_Intent (Need-based, Impulsive, Planned, Wants-based)

---

## 📁 Project Structure

```
innofusion-hackathon/
│
├── 📄 Propmt.md                                          # Requirements specification
├── 📄 README.md                                          # General project info
├── 📄 PROJECT_ARCHITECTURE.md                            # This document
│
├── 📊 Ecommerce_Consumer_Behavior_Analysis_Data.csv      # Raw dataset (27 columns)
│
├── 🐍 section_4_ml_notebook.py                           # Main ML pipeline (Colab-ready)
│
├── 📂 outputs/                                           # Generated artifacts
│   ├── model_comparison.png                              # Model accuracy comparison chart
│   ├── confusion_matrix.png                              # Best model confusion matrix
│   └── feature_importance.png                            # Feature importance analysis
│
└── 📂 .vscode/                                           # VS Code settings (optional)
```

---

## 🏗️ Architecture Overview

### **Architecture Type**: Sequential ML Pipeline

This project implements a **linear, end-to-end machine learning pipeline** designed for Google Colab execution. The architecture follows a sequential flow without complex abstractions, making it beginner-friendly and easy to debug.

---

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW DATA LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  File: Ecommerce_Consumer_Behavior_Analysis_Data.csv            │
│  Rows: ~N samples                                               │
│  Cols: 27 features (mixed types)                                │
│        ├── 3 columns to drop                                    │
│        ├── 1 column to clean (Purchase_Amount: "$XX" → float)  │
│        ├── 2 boolean columns (TRUE/FALSE → 1/0)                │
│        ├── 13 categorical columns (to encode)                   │
│        └── 1 target column (Purchase_Intent)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  ✓ Drop: Customer_ID, Location, Time_of_Purchase               │
│  ✓ Clean: Purchase_Amount (remove "$" and spaces)              │
│  ✓ Convert: Discount_Used, Customer_Loyalty_Program_Member     │
│  ✓ Encode: 13 categorical features using LabelEncoder          │
│  ✓ Encode: Target variable (save encoder for inverse mapping)  │
│                                                                 │
│  Output: Clean DataFrame (24 columns, all numeric)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  Feature Selection: 11 key features                            │
│  ├── Numerical (6): Age, Purchase_Amount,                      │
│  │                   Frequency_of_Purchase,                    │
│  │                   Customer_Satisfaction, Brand_Loyalty,     │
│  │                   Product_Rating,                           │
│  │                   Time_Spent_on_Product_Research(hours)     │
│  │                                                             │
│  └── Categorical Encoded (4): Discount_Sensitivity,            │
│                                Income_Level,                    │
│                                Engagement_with_Ads,             │
│                                Social_Media_Influence           │
│                                                                 │
│  Target: Purchase_Intent (4 classes)                           │
│  Split: 80% train / 20% test (stratified)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┏━━━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━┓  │
│  ┃   Logistic      ┃  ┃  Random Forest  ┃  ┃    XGBoost    ┃  │
│  ┃   Regression    ┃  ┃   Classifier    ┃  ┃   Classifier  ┃  │
│  ┣━━━━━━━━━━━━━━━━━┫  ┣━━━━━━━━━━━━━━━━━┫  ┣━━━━━━━━━━━━━━━┫  │
│  ┃ max_iter: 1000  ┃  ┃ n_estimators:   ┃  ┃ n_estimators: ┃  │
│  ┃ random_state:42 ┃  ┃ 100             ┃  ┃ 200           ┃  │
│  ┃                 ┃  ┃ max_depth: 10   ┃  ┃ max_depth: 4  ┃  │
│  ┃ Purpose:        ┃  ┃ class_weight:   ┃  ┃ learning_rate:┃  │
│  ┃ Baseline model  ┃  ┃ balanced        ┃  ┃ 0.1           ┃  │
│  ┃                 ┃  ┃ random_state:42 ┃  ┃ subsample:0.8 ┃  │
│  ┃                 ┃  ┃                 ┃  ┃ random_state: ┃  │
│  ┃                 ┃  ┃ Purpose:        ┃  ┃ 42            ┃  │
│  ┃                 ┃  ┃ Feature         ┃  ┃               ┃  │
│  ┃                 ┃  ┃ importance      ┃  ┃ Purpose:      ┃  │
│  ┃                 ┃  ┃ analysis        ┃  ┃ Best accuracy ┃  │
│  ┗━━━━━━━━━━━━━━━━━┛  ┗━━━━━━━━━━━━━━━━━┛  ┗━━━━━━━━━━━━━━━┛  │
│                                                                 │
│  → All models trained on same train/test split                 │
│  → Predictions generated for test set                          │
│  → Accuracies stored: lr_accuracy, rf_accuracy, xgb_accuracy   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL EVALUATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Compare Accuracies                                          │
│     └── Identify best model automatically                       │
│                                                                 │
│  2. Detailed Evaluation of Best Model                           │
│     ├── Confusion Matrix (with actual class names)             │
│     ├── Classification Report (precision/recall/F1)            │
│     └── 5-Fold Cross Validation                                │
│                                                                 │
│  3. Feature Importance Analysis                                 │
│     └── Extract from Random Forest (regardless of best model)   │
│                                                                 │
│  Metrics Tracked:                                               │
│  ├── Test Accuracy                                              │
│  ├── Cross-Validation Mean ± Std                               │
│  ├── Per-Class Precision, Recall, F1-Score                     │
│  └── Feature Importance Scores                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              VISUALIZATION & INSIGHTS LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Chart 1: model_comparison.png                              │
│     ├── Type: Horizontal bar chart                             │
│     ├── Data: 3 model accuracies                               │
│     ├── Colors: ['#4C72B0', '#55A868', '#C44E52']              │
│     └── Features: Dashed line at max accuracy                  │
│                                                                 │
│  📊 Chart 2: confusion_matrix.png                              │
│     ├── Type: Seaborn heatmap                                  │
│     ├── Data: Best model predictions vs actual                 │
│     ├── Color: 'Blues' colormap                                │
│     └── Labels: Actual class names (inverse transformed)       │
│                                                                 │
│  📊 Chart 3: feature_importance.png                            │
│     ├── Type: Horizontal bar chart                             │
│     ├── Data: Random Forest feature importances                │
│     ├── Color: '#4C72B0'                                       │
│     └── Sort: Descending by importance                         │
│                                                                 │
│  📋 Business Summary Box                                        │
│     ├── Best model name and accuracy                           │
│     ├── Cross-validation scores                                │
│     ├── Top 3 features                                         │
│     └── Auto-generated business insights                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Files Generated:                                               │
│  ├── outputs/model_comparison.png                              │
│  ├── outputs/confusion_matrix.png                              │
│  └── outputs/feature_importance.png                            │
│                                                                 │
│  Console Outputs:                                               │
│  ├── Class mapping (0→Impulsive, 1→Need-based, etc.)          │
│  ├── Train/test split sizes                                    │
│  ├── Model accuracies                                           │
│  ├── Classification report (precision/recall/F1)               │
│  ├── Cross-validation results                                  │
│  └── Business summary box                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technical Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.8+ | Core programming |
| **Notebook** | Google Colab | Latest | Execution environment |
| **Data Processing** | pandas | Latest | Data manipulation |
| **Numerical Computing** | numpy | Latest | Array operations |
| **Visualization** | matplotlib | Latest | Chart generation |
| **Statistical Viz** | seaborn | Latest | Heatmaps & advanced plots |
| **ML Framework** | scikit-learn | Latest | Classical ML models |
| **Gradient Boosting** | xgboost | Latest | XGBoost classifier |

---

## 🎯 Design Principles

### 1. **Simplicity First**
- No custom classes or complex abstractions
- Sequential code blocks (cell-by-cell execution)
- Beginner-friendly comments throughout

### 2. **Reproducibility**
- `random_state=42` in all stochastic operations
- Fixed train/test split ratio (80/20)
- Stratified sampling for class balance

### 3. **Modularity**
- Each section is independent
- Clear section headers with visual separators
- Success messages after each section

### 4. **Visual Clarity**
- Minimum figure size: (10, 6)
- Consistent color schemes
- Descriptive titles and axis labels
- Saved as high-quality PNG files

### 5. **Business Focus**
- Auto-generated insights from model results
- Clear summary box with key metrics
- Feature importance analysis for decision-making

---

## 📊 Data Architecture

### **Input Schema**
```
Dataset: Ecommerce_Consumer_Behavior_Analysis_Data.csv
├── Total Columns: 27
├── Total Rows: ~N (varies by dataset version)
│
├── Columns to Drop (3):
│   ├── Customer_ID        : Unique identifier (no predictive value)
│   ├── Location           : Text field (too granular)
│   └── Time_of_Purchase   : Date string (timing not in scope)
│
├── Columns to Clean (1):
│   └── Purchase_Amount    : "$XXX.XX " → float (remove $ and spaces)
│
├── Columns to Convert (2):
│   ├── Discount_Used                      : "TRUE"/"FALSE" → 1/0
│   └── Customer_Loyalty_Program_Member    : "TRUE"/"FALSE" → 1/0
│
├── Categorical Columns to Encode (13):
│   ├── Gender
│   ├── Income_Level
│   ├── Marital_Status
│   ├── Education_Level
│   ├── Occupation
│   ├── Purchase_Category
│   ├── Purchase_Channel
│   ├── Social_Media_Influence
│   ├── Discount_Sensitivity
│   ├── Engagement_with_Ads
│   ├── Device_Used_for_Shopping
│   ├── Payment_Method
│   └── Shipping_Preference
│
├── Target Variable (1):
│   └── Purchase_Intent     : (Need-based, Impulsive, Planned, Wants-based)
│
└── Other Features (7 numerical):
    ├── Age
    ├── Frequency_of_Purchase
    ├── Brand_Loyalty
    ├── Product_Rating
    ├── Time_Spent_on_Product_Research(hours)
    ├── Return_Rate
    ├── Customer_Satisfaction
    └── Time_to_Decision
```

### **Feature Selection Strategy**
Selected **11 features** covering entire customer journey:

1. **Demographics** (2 features)
   - Age
   - Income_Level (encoded)

2. **Financial Behavior** (2 features)
   - Purchase_Amount
   - Discount_Sensitivity (encoded)

3. **Engagement Metrics** (3 features)
   - Social_Media_Influence (encoded)
   - Engagement_with_Ads (encoded)
   - Time_Spent_on_Product_Research(hours)

4. **Purchase Behavior** (2 features)
   - Frequency_of_Purchase
   - Brand_Loyalty

5. **Satisfaction Metrics** (2 features)
   - Customer_Satisfaction
   - Product_Rating

---

## 🤖 Model Architecture

### **Model 1: Logistic Regression**
```python
LogisticRegression(
    max_iter=1000,        # Sufficient for convergence
    random_state=42       # Reproducibility
)
```
- **Purpose**: Baseline model
- **Complexity**: Low
- **Interpretability**: High
- **Expected Accuracy**: 45-55%

### **Model 2: Random Forest**
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Prevent overfitting
    random_state=42,       # Reproducibility
    class_weight='balanced' # Handle class imbalance
)
```
- **Purpose**: Feature importance analysis
- **Complexity**: Medium
- **Interpretability**: Medium (via feature importance)
- **Expected Accuracy**: 60-70%

### **Model 3: XGBoost**
```python
XGBClassifier(
    n_estimators=200,      # More trees = better performance
    max_depth=4,           # Shallow trees (prevent overfit)
    learning_rate=0.1,     # Standard learning rate
    subsample=0.8,         # 80% of samples per tree
    colsample_bytree=0.8,  # 80% of features per tree
    use_label_encoder=False, # Suppress warning
    eval_metric='mlogloss', # Multiclass log loss
    random_state=42        # Reproducibility
)
```
- **Purpose**: Best accuracy (likely winner)
- **Complexity**: High
- **Interpretability**: Low
- **Expected Accuracy**: 65-75%+

---

## 📈 Evaluation Metrics

### **Primary Metric**
- **Accuracy**: Proportion of correct predictions
  - Used for model comparison
  - Simple and interpretable

### **Secondary Metrics**
- **Precision**: True Positives / (True Positives + False Positives)
  - Per-class precision reported
- **Recall**: True Positives / (True Positives + False Negatives)
  - Per-class recall reported
- **F1-Score**: Harmonic mean of precision and recall
  - Balanced metric for each class

### **Validation Strategy**
- **Train-Test Split**: 80/20 with stratification
- **Cross-Validation**: 5-fold on best model
  - Reports mean ± standard deviation
  - Checks model stability

---

## 🎨 Visualization Standards

### **Chart 1: Model Comparison**
```
Type: Horizontal Bar Chart
Size: (10, 6)
Colors: ['#4C72B0', '#55A868', '#C44E52']
        (Blue, Green, Red)
X-axis: Accuracy (0.0 to 1.0)
Y-axis: Model Names
Special: Dashed vertical line at max accuracy
Labels: Accuracy values displayed on bars
```

### **Chart 2: Confusion Matrix**
```
Type: Seaborn Heatmap
Size: (10, 6)
Colormap: 'Blues'
Annot: True (show counts in cells)
Fmt: 'd' (integer format)
Labels: Actual class names (inverse transformed)
Title: "Confusion Matrix - [Best Model Name]"
```

### **Chart 3: Feature Importance**
```
Type: Horizontal Bar Chart
Size: (10, 6)
Color: '#4C72B0' (Blue)
Sort: Descending by importance score
X-axis: Feature Importance Score
Y-axis: Feature Names
Labels: Importance values displayed on bars
```

---

## 🔐 Code Quality Standards

### **Comments**
```python
# ── SECTION X.X: TITLE ────────────────────────────────
# Brief description of what this section does

# Inline comment for important operations
variable = operation()  # Brief explanation if needed

print("✅ Section X.X complete")
```

### **Naming Conventions**
- **Variables**: `snake_case` (e.g., `lr_accuracy`)
- **Constants**: `UPPER_SNAKE_CASE` (if any)
- **Descriptive**: `best_model` not `bm`

### **Error Handling**
- Assume clean data (hackathon context)
- No try-except blocks (keep code simple)
- Let errors surface for debugging

---

## 🚀 Execution Flow

### **Sequential Execution (10 Steps)**
```
Step 1: Install Dependencies
  └── !pip install xgboost -q

Step 2: Import Libraries
  └── All imports at top

Step 3-12: Execute Sections 4.1 through 4.10
  ├── Each section prints success message
  ├── Sections are independent
  └── Can be run as separate Colab cells

Final: Complete Message
  └── "🎉 Machine Learning Pipeline Complete!"
```

### **Estimated Runtime**
- **Data Loading**: ~1-2 seconds
- **Preprocessing**: ~2-3 seconds
- **Model Training**: ~10-30 seconds (all 3 models)
- **Evaluation/Visualization**: ~5-10 seconds
- **Total**: ~20-50 seconds (single Colab run)

---

## 📤 Deliverables Checklist

### **Code Deliverable**
- [x] `section_4_ml_notebook.py` (Colab-ready Python script)

### **Visual Outputs**
- [x] `model_comparison.png` (3 models compared)
- [x] `confusion_matrix.png` (best model evaluation)
- [x] `feature_importance.png` (RF importance analysis)

### **Console Outputs**
- [x] Class mapping printed
- [x] Train/test sizes printed
- [x] Model accuracies printed
- [x] Classification report printed
- [x] Cross-validation results printed
- [x] Business summary box printed

---

## 🎯 Success Criteria

✅ **Functionality**
- All 10 sections execute without errors
- Models train and predict successfully
- Charts save to disk correctly

✅ **Accuracy**
- At least one model achieves >60% accuracy
- XGBoost expected to win (~65-75%)

✅ **Reproducibility**
- Same results on every run (random_state=42)
- Cross-validation shows model stability

✅ **Business Value**
- Feature importance identifies key drivers
- Insights are actionable and clear
- Summary box provides complete overview

---

## 🔮 Future Enhancements (Post-Hackathon)

### **Model Improvements**
- Hyperparameter tuning (GridSearchCV)
- Try neural networks (TensorFlow/PyTorch)
- Ensemble methods (voting classifier)

### **Feature Engineering**
- Create interaction features
- Polynomial features for non-linearity
- Feature scaling (StandardScaler)

### **Advanced Analysis**
- SHAP values for model interpretability
- ROC curves for each class (OvR)
- Learning curves to detect overfitting

### **Production Readiness**
- Save best model (pickle/joblib)
- Create prediction API (Flask/FastAPI)
- Deploy to cloud (AWS/GCP/Azure)

---

## 📝 Author Notes

**Development Date**: February 2026  
**Hackathon**: Innofusion'26  
**Section**: 4 - Machine Learning  
**Environment**: Google Colab  
**Coding Style**: Beginner-friendly, sequential, well-commented

---

## 📞 Support

For questions or issues:
1. Check section-by-section outputs
2. Verify dataset file name matches code
3. Ensure Google Colab has sufficient resources
4. Review error messages for specific issues

---

**End of Architecture Document**
