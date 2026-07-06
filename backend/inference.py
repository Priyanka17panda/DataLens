def detect_ml_task(df, column_types):
    possible_target = None
    problem_type = "Unknown"
    reason = ""

    target_keywords = [
        "target",
        "label",
        "class",
        "output",
        "y",
        "churn",
        "default",
        "fraud",
        "survived",
        "price",
        "sales",
        "status",
        "result"
    ]

    # Step 1: try to find target-like column by name
    for col in df.columns:
        if col.lower() in target_keywords:
            possible_target = col
            break

    # Step 2: if no keyword match, use last column as fallback
    if possible_target is None:
        possible_target = df.columns[-1]

    unique_values = df[possible_target].nunique(dropna=True)

    # NLP check first
    if len(column_types["text"]) > 0:
        problem_type = "NLP / Text Analysis"
        reason = "Text columns detected in the dataset."

    # Time series check next
    elif len(column_types["datetime"]) > 0:
        problem_type = "Time Series Forecasting"
        reason = "Datetime column detected in the dataset."

    # Classification
    elif unique_values <= 10:
        problem_type = "Classification"
        reason = f"{possible_target} has {unique_values} unique values, which suggests a classification problem."

    # Regression
    elif possible_target in column_types["numerical"] and unique_values > 10:
        problem_type = "Regression"
        reason = f"{possible_target} appears to be a continuous numerical target."

    # Clustering fallback
    else:
        problem_type = "Clustering / Unsupervised Learning"
        reason = "No strongly defined supervised target pattern was detected."

    return {
        "problem_type": problem_type,
        "possible_target": possible_target,
        "reason": reason
    }