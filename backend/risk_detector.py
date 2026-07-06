def detect_dataset_risks(df, dataset_profile, column_types, task_info, target_analysis):
    risks = []

    rows = dataset_profile["rows"]
    cols = dataset_profile["columns"]
    missing_percentage = dataset_profile["missing_percentage"]
    id_like_count = dataset_profile["id_like_count"]
    numerical_count = dataset_profile["numerical_count"]
    categorical_count = dataset_profile["categorical_count"]
    high_cardinality_cols = dataset_profile["high_cardinality_cols"]

    problem_type = task_info["problem_type"]

    # Missing data risks
    if missing_percentage > 30:
        risks.append("High missing value percentage detected. Missing data handling will be a major challenge.")
    elif missing_percentage > 10:
        risks.append("Moderate missing value percentage detected. Careful imputation will be required.")

    # Imbalance risk
    if target_analysis["is_imbalanced"]:
        risks.append("The target appears imbalanced. Model evaluation should focus on metrics like F1-score, recall, precision, and ROC-AUC instead of accuracy alone.")

    # ID columns
    if id_like_count > 0:
        risks.append("ID-like columns are present. These can introduce noise or leakage if used directly in modeling.")

    # High-cardinality categoricals
    if len(high_cardinality_cols) > 0:
        risks.append("High-cardinality categorical columns detected. Naive one-hot encoding may create too many sparse features.")

    # Very small dataset
    if rows < 500:
        risks.append("The dataset is very small. Model performance estimates may be unstable, and simpler models may generalize better.")

    # Feature-to-sample ratio
    if rows > 0 and cols / rows > 0.5:
        risks.append("The dataset has a high feature-to-row ratio, which may increase overfitting risk.")

    # Too few numerical columns
    if numerical_count == 0 and problem_type in ["Classification", "Regression"]:
        risks.append("No numerical features were detected for a supervised task. Modeling may rely heavily on categorical encoding.")

    # Too few categorical/numerical diversity
    if numerical_count < 2 and categorical_count < 2:
        risks.append("The dataset has very limited feature diversity, which may reduce predictive modeling potential.")

    # Time series split warning
    if problem_type == "Time Series Forecasting":
        risks.append("For time series problems, random train-test splitting should be avoided because it breaks temporal order.")

    # NLP warning
    if problem_type == "NLP / Text Analysis":
        risks.append("For text-heavy datasets, preprocessing quality and text representation choice will strongly affect performance.")

    # If nothing serious found
    if len(risks) == 0:
        risks.append("No major structural risks were detected from the current dataset profile, though deeper EDA is still recommended.")

    return risks