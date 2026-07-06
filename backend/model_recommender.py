def recommend_models(problem_type, dataset_profile, column_types, target_analysis):
    models = []
    reasons = []

    rows = dataset_profile["rows"]
    categorical_count = dataset_profile["categorical_count"]
    missing_percentage = dataset_profile["missing_percentage"]
    high_cardinality_cols = dataset_profile["high_cardinality_cols"]
    is_imbalanced = target_analysis["is_imbalanced"]

    if problem_type == "Classification":
        models.append("Logistic Regression")
        reasons.append("Good baseline model for classification and easy to interpret.")

        models.append("Random Forest Classifier")
        reasons.append("Works well on structured tabular data and captures non-linear patterns.")

        models.append("XGBoost Classifier")
        reasons.append("Strong performance on structured/tabular datasets.")

        if categorical_count >= 5:
            models.append("CatBoost Classifier")
            reasons.append("Useful when the dataset has many categorical columns.")

        if rows > 50000:
            models.append("LightGBM Classifier")
            reasons.append("Efficient for large datasets and faster to train than many alternatives.")

        if is_imbalanced:
            models.append("Balanced Random Forest / Class-Weighted Models")
            reasons.append("Useful because the target appears imbalanced.")

        if rows < 20000 and categorical_count < 20:
            models.append("Support Vector Machine")
            reasons.append("Can work well for smaller structured datasets.")

        if missing_percentage > 10:
            models.append("Tree-Based Boosting Models")
            reasons.append("Recommended because the dataset has notable missing values and boosting models usually perform well on such structured data.")

        if len(high_cardinality_cols) > 0:
            models.append("CatBoost with Encoded High-Cardinality Features")
            reasons.append("Useful because some categorical columns have high cardinality.")

    elif problem_type == "Regression":
        models.append("Linear Regression")
        reasons.append("Simple baseline for regression tasks.")

        models.append("Random Forest Regressor")
        reasons.append("Captures non-linear relationships in tabular data.")

        models.append("XGBoost Regressor")
        reasons.append("Strong predictive performance on structured data.")

        if categorical_count >= 5:
            models.append("CatBoost Regressor")
            reasons.append("Useful when the dataset includes many categorical features.")

        if rows > 50000:
            models.append("LightGBM Regressor")
            reasons.append("Efficient for large-scale structured data.")

        if missing_percentage > 10:
            models.append("Tree-Based Boosting Regressors")
            reasons.append("Recommended because the dataset has notable missing values.")

        if len(high_cardinality_cols) > 0:
            models.append("CatBoost Regressor with Encoded High-Cardinality Features")
            reasons.append("Useful because some categorical columns have high cardinality.")

    elif problem_type == "NLP / Text Analysis":
        models.extend([
            "TF-IDF + Logistic Regression",
            "Naive Bayes",
            "BERT"
        ])
        reasons.extend([
            "Strong baseline for text classification.",
            "Effective for sparse text features.",
            "Powerful contextual model for advanced NLP tasks."
        ])

    elif problem_type == "Time Series Forecasting":
        models.extend([
            "ARIMA",
            "Prophet",
            "XGBoost with Lag Features",
            "LSTM"
        ])
        reasons.extend([
            "Classical baseline for time series forecasting.",
            "Useful for trend and seasonality.",
            "Good for engineered time-based tabular features.",
            "Useful for more advanced sequential modeling."
        ])

    else:
        models.extend([
            "K-Means",
            "DBSCAN",
            "Hierarchical Clustering"
        ])
        reasons.extend([
            "Common baseline for segmentation.",
            "Useful for density-based clusters.",
            "Helpful for hierarchical grouping."
        ])

    final_models = []
    final_reasons = []
    seen = set()

    for model, reason in zip(models, reasons):
        if model not in seen:
            seen.add(model)
            final_models.append(model)
            final_reasons.append(reason)

    return {
        "models": final_models,
        "reasons": final_reasons
    }