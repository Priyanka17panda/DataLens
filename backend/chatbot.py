def answer_user_query(
    user_query,
    basic_info,
    dataset_profile,
    column_types,
    task_info,
    target_analysis,
    preprocessing_steps,
    feature_insights,
    detected_risks,
    models,
    summary_text,
    null_info
):
    query = user_query.strip().lower()

    if not query:
        return "Please type a question about the uploaded dataset."

    if "summary" in query or "overview" in query or "about dataset" in query:
        return summary_text

    if "rows" in query and "columns" in query:
        return f"The dataset has {basic_info['rows']} rows and {basic_info['columns']} columns."

    if "how many rows" in query:
        return f"The dataset contains {basic_info['rows']} rows."

    if "how many columns" in query:
        return f"The dataset contains {basic_info['columns']} columns."

    if "problem type" in query or "classification" in query or "regression" in query or "clustering" in query or "time series" in query or "nlp" in query:
        return (
            f"The detected ML problem type is {task_info['problem_type']}. "
            f"Reason: {task_info['reason']}"
        )

    if "target" in query:
        return (
            f"The likely target column is '{task_info['possible_target']}'. "
            f"Target analysis suggests {target_analysis['unique_values']} unique values."
        )

    if "imbalanced" in query or "imbalance" in query or "class balance" in query:
        return (
            f"Is imbalanced: {target_analysis['is_imbalanced']}. "
            f"Class balance: {target_analysis['class_balance']}"
        )

    if "null" in query or "missing" in query:
        top_nulls = null_info[null_info["Null Count"] > 0].head(10)

        if top_nulls.empty:
            return "No missing values were detected in the dataset."

        missing_lines = []
        for _, row in top_nulls.iterrows():
            missing_lines.append(
                f"{row['Column']}: {int(row['Null Count'])} missing ({row['Null Percentage']}%)"
            )

        return "Top columns with missing values:\n" + "\n".join(missing_lines)

    if "column type" in query or "categorical" in query or "numerical" in query or "text" in query or "datetime" in query:
        return (
            f"Numerical columns: {column_types['numerical']}\n"
            f"Categorical columns: {column_types['categorical']}\n"
            f"Text columns: {column_types['text']}\n"
            f"Datetime columns: {column_types['datetime']}\n"
            f"Boolean columns: {column_types['boolean']}\n"
            f"ID-like columns: {column_types['id_like']}"
        )

    if "preprocessing" in query or "cleaning" in query or "what should i do first" in query:
        return "Recommended preprocessing steps:\n" + "\n".join(
            [f"- {step}" for step in preprocessing_steps]
        )

    if "feature" in query or "important feature" in query or "correlated" in query:
        return (
            f"Top correlated numerical features: {feature_insights['top_correlated_features']}\n"
            f"Low information features: {feature_insights['low_information_features']}\n"
            f"Drop candidate features: {feature_insights['drop_candidate_features']}"
        )

    if "risk" in query or "warning" in query or "problem in dataset" in query:
        return "Detected risks/warnings:\n" + "\n".join(
            [f"- {risk}" for risk in detected_risks]
        )

    if "model" in query or "algorithm" in query or "which model" in query:
        model_lines = [f"- {model}" for model in models]
        return "Recommended models for this dataset:\n" + "\n".join(model_lines)

    if "dataset size" in query or "small" in query or "medium" in query or "large" in query:
        return (
            f"The dataset is categorized as {dataset_profile['size_label']} "
            f"with {dataset_profile['rows']} rows and {dataset_profile['columns']} columns."
        )

    return (
        "I can answer questions about dataset summary, target column, problem type, "
        "missing values, preprocessing, risks, feature insights, dataset size, and recommended models."
    )