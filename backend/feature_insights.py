import pandas as pd


def get_feature_insights(df, column_types, task_info):
    problem_type = task_info["problem_type"]
    target_col = task_info["possible_target"]

    insights = {
        "top_correlated_features": [],
        "low_information_features": [],
        "drop_candidate_features": [],
        "notes": []
    }

    # ID-like columns are usually drop candidates
    insights["drop_candidate_features"].extend(column_types["id_like"])

    # Low-information categorical columns: only 1 unique value
    for col in column_types["categorical"]:
        if df[col].dropna().nunique() <= 1:
            insights["low_information_features"].append(col)

    # Only do correlation-based feature ranking for supervised numeric targets
    if problem_type in ["Classification", "Regression"] and target_col in df.columns:
        if target_col in column_types["numerical"] or target_col in column_types["boolean"]:
            numeric_candidates = [
                col for col in column_types["numerical"]
                if col != target_col and col in df.columns
            ]

            if len(numeric_candidates) > 0:
                corr_df = df[numeric_candidates + [target_col]].copy()

                # safe numeric conversion
                corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

                corr_matrix = corr_df.corr(numeric_only=True)

                if target_col in corr_matrix.columns:
                    target_corr = corr_matrix[target_col].drop(labels=[target_col], errors="ignore")
                    target_corr = target_corr.abs().sort_values(ascending=False)

                    top_features = target_corr.head(5).index.tolist()
                    insights["top_correlated_features"] = top_features

                    if len(top_features) > 0:
                        insights["notes"].append(
                            f"Top numerical features most associated with '{target_col}' were estimated using correlation analysis."
                        )
                else:
                    insights["notes"].append(
                        "Could not compute target correlation reliably for the selected target column."
                    )
            else:
                insights["notes"].append(
                    "No suitable numerical predictors were found for correlation-based feature insight generation."
                )
        else:
            insights["notes"].append(
                "The detected target is not numerical/boolean, so correlation-based feature analysis was limited."
            )

    else:
        insights["notes"].append(
            "Feature insight estimation is strongest for supervised classification/regression datasets."
        )

    if len(insights["drop_candidate_features"]) > 0:
        insights["notes"].append(
            "ID-like columns were marked as drop candidates because they usually do not help modeling directly."
        )

    if len(insights["low_information_features"]) > 0:
        insights["notes"].append(
            "Some columns appear to have very low information because they contain only one unique value."
        )

    return insights