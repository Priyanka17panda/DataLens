def generate_dataset_summary(dataset_profile, column_types, task_info, target_analysis):
    rows = dataset_profile["rows"]
    cols = dataset_profile["columns"]
    size_label = dataset_profile["size_label"]
    missing_percentage = dataset_profile["missing_percentage"]

    numerical_count = dataset_profile["numerical_count"]
    categorical_count = dataset_profile["categorical_count"]
    text_count = dataset_profile["text_count"]
    datetime_count = dataset_profile["datetime_count"]
    id_like_count = dataset_profile["id_like_count"]

    problem_type = task_info["problem_type"]
    target_col = task_info["possible_target"]

    summary_parts = []

    # Basic dataset overview
    summary_parts.append(
        f"This dataset contains {rows} rows and {cols} columns, so it can be considered a {size_label.lower()} dataset."
    )

    # Feature composition
    summary_parts.append(
        f"It includes {numerical_count} numerical columns and {categorical_count} categorical columns."
    )

    if text_count > 0:
        summary_parts.append(
            f"There are also {text_count} text-based columns, which may support NLP-related analysis."
        )

    if datetime_count > 0:
        summary_parts.append(
            f"The presence of {datetime_count} datetime columns suggests possible time-based analysis or forecasting opportunities."
        )

    if id_like_count > 0:
        summary_parts.append(
            f"The dataset also contains {id_like_count} ID-like columns, which are usually not useful as predictive features directly."
        )

    # Missing value understanding
    if missing_percentage == 0:
        summary_parts.append(
            "The dataset has no missing values, which simplifies preprocessing."
        )
    elif missing_percentage < 5:
        summary_parts.append(
            f"The dataset has a low missing value percentage of {missing_percentage}%, so only limited imputation may be needed."
        )
    elif missing_percentage < 20:
        summary_parts.append(
            f"The dataset has a moderate missing value percentage of {missing_percentage}%, so careful preprocessing and imputation will be important."
        )
    else:
        summary_parts.append(
            f"The dataset has a high missing value percentage of {missing_percentage}%, so missing value handling will be a major preprocessing step."
        )

    # Problem type
    summary_parts.append(
        f"Based on the detected structure, the dataset appears suitable for a {problem_type.lower()} problem."
    )

    if target_col:
        summary_parts.append(
            f"The likely target column is '{target_col}'."
        )

    # Target analysis
    if target_analysis["target_found"] and target_analysis["unique_values"] is not None:
        summary_parts.append(
            f"The target column has {target_analysis['unique_values']} unique values."
        )

    if target_analysis["is_imbalanced"]:
        summary_parts.append(
            "The target distribution appears imbalanced, so imbalance-aware methods, resampling, or class-weighted models should be considered."
        )

    # Final recommendation tone
    if problem_type == "Classification":
        summary_parts.append(
            "For this type of structured dataset, strong candidate models include Logistic Regression as a baseline, and advanced tree-based methods such as Random Forest, XGBoost, LightGBM, and CatBoost."
        )
    elif problem_type == "Regression":
        summary_parts.append(
            "For this dataset, baseline regression models can be used first, followed by stronger tree-based regression methods such as Random Forest Regressor, XGBoost Regressor, LightGBM Regressor, or CatBoost Regressor."
        )
    elif problem_type == "NLP / Text Analysis":
        summary_parts.append(
            "Since text data is present, common approaches include TF-IDF based models, Naive Bayes, Logistic Regression, and transformer-based methods like BERT."
        )
    elif problem_type == "Time Series Forecasting":
        summary_parts.append(
            "Since the dataset includes temporal structure, time series models such as ARIMA, Prophet, lag-based boosting models, or LSTM may be suitable."
        )
    else:
        summary_parts.append(
            "Since no strong supervised target pattern is detected, exploratory analysis and unsupervised methods such as clustering may be appropriate."
        )

    return " ".join(summary_parts)