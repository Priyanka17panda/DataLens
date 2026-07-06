def recommend_preprocessing(dataset_profile, column_types, task_info, target_analysis):
    steps = []

    missing_percentage = dataset_profile["missing_percentage"]
    categorical_count = dataset_profile["categorical_count"]
    numerical_count = dataset_profile["numerical_count"]
    text_count = dataset_profile["text_count"]
    datetime_count = dataset_profile["datetime_count"]
    id_like_count = dataset_profile["id_like_count"]
    high_cardinality_cols = dataset_profile["high_cardinality_cols"]

    problem_type = task_info["problem_type"]
    target_col = task_info["possible_target"]

    if missing_percentage > 0:
        if missing_percentage < 5:
            steps.append("Handle missing values using simple imputation, since the overall missing percentage is low.")
        elif missing_percentage < 20:
            steps.append("Apply careful missing value imputation because the dataset has a moderate amount of missing data.")
        else:
            steps.append("Missing value handling should be a major preprocessing step because the dataset has a high missing percentage.")

    if id_like_count > 0:
        steps.append("Remove or exclude ID-like columns from modeling, since they usually do not provide predictive value.")

    if categorical_count > 0:
        steps.append("Encode categorical features using techniques such as one-hot encoding, label encoding, or target/frequency encoding depending on the model.")

    if len(high_cardinality_cols) > 0:
        steps.append("Use special handling for high-cardinality categorical columns, such as target encoding, frequency encoding, or CatBoost-style handling.")

    if numerical_count > 0:
        steps.append("Check numerical features for outliers, skewness, and scaling requirements before training certain models.")

    if text_count > 0:
        steps.append("Clean text columns by lowercasing, removing noise, tokenizing, and converting text into numerical features such as TF-IDF or embeddings.")

    if datetime_count > 0:
        steps.append("Extract useful time-based features such as year, month, day, weekday, lag features, or rolling statistics from datetime columns.")

    if target_analysis["is_imbalanced"]:
        steps.append("Handle target imbalance using class weights, SMOTE, undersampling, or imbalance-aware algorithms.")

    if problem_type in ["Classification", "Regression"]:
        steps.append(f"Separate the target column '{target_col}' from the input features before model training.")
        steps.append("Perform train-test split before final modeling to evaluate generalization properly.")

    if problem_type == "Classification":
        steps.append("Use stratified train-test split if possible, so class proportions remain stable across train and test sets.")

    if problem_type == "Regression":
        steps.append("Check the target variable distribution and consider transformation if the target is highly skewed.")

    if problem_type == "Time Series Forecasting":
        steps.append("Do not randomly shuffle the data; preserve temporal order while splitting training and validation sets.")

    if problem_type == "NLP / Text Analysis":
        steps.append("Convert raw text into machine-readable features using TF-IDF, bag-of-words, embeddings, or transformer tokenization.")

    if problem_type == "Clustering / Unsupervised Learning":
        steps.append("Standardize numerical features before clustering so one feature scale does not dominate the others.")

    return steps