import streamlit as st
from analyzer import (
    load_data,
    get_basic_info,
    get_column_names,
    get_data_types,
    get_null_info,
    detect_column_types,
    get_dataset_profile,
    get_target_analysis
)
from inference import detect_ml_task
from model_recommender import recommend_models
from summarizer import generate_dataset_summary
from preprocessor_recommender import recommend_preprocessing
from feature_insights import get_feature_insights
from risk_detector import detect_dataset_risks
from visualizer import plot_missing_values, plot_target_distribution, plot_correlation_heatmap
from chatbot import answer_user_query

st.set_page_config(
    page_title="Smart Dataset Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("Smart Dataset Assistant")
st.caption("Upload a dataset and start understanding it instantly.")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Dataset Structure",
        "ML Insights",
        "EDA Visualizations",
        "Chatbot"
    ]
)

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)

        basic_info = get_basic_info(df)
        column_names = get_column_names(df)
        data_types = get_data_types(df)
        null_info = get_null_info(df)
        column_types = detect_column_types(df)
        dataset_profile = get_dataset_profile(df, column_types)

        task_info = detect_ml_task(df, column_types)
        target_analysis = get_target_analysis(df, task_info["possible_target"])

        recommendation_output = recommend_models(
            task_info["problem_type"],
            dataset_profile,
            column_types,
            target_analysis
        )
        models = recommendation_output["models"]
        reasons = recommendation_output["reasons"]

        summary_text = generate_dataset_summary(
            dataset_profile,
            column_types,
            task_info,
            target_analysis
        )

        preprocessing_steps = recommend_preprocessing(
            dataset_profile,
            column_types,
            task_info,
            target_analysis
        )

        feature_insights = get_feature_insights(df, column_types, task_info)

        detected_risks = detect_dataset_risks(
            df,
            dataset_profile,
            column_types,
            task_info,
            target_analysis
        )

        missing_fig = plot_missing_values(df)
        target_fig = plot_target_distribution(df, task_info["possible_target"])
        corr_fig = plot_correlation_heatmap(df, column_types)

        st.success("Dataset uploaded successfully.")

        # ===============================
        # OVERVIEW
        # ===============================
        if section == "Overview":
            st.header("Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", basic_info["rows"])
            col2.metric("Columns", basic_info["columns"])
            col3.metric("Duplicate Rows", basic_info["duplicate_rows"])
            col4.metric("Memory (KB)", basic_info["memory_usage_kb"])

            with st.expander("AI Generated Dataset Summary", expanded=True):
                st.write(summary_text)

            with st.expander("Target Information", expanded=True):
                st.write("Problem Type:", task_info["problem_type"])
                st.write("Target Column:", task_info["possible_target"])
                st.write("Target Found:", target_analysis["target_found"])
                st.write("Unique Values in Target:", target_analysis["unique_values"])
                st.write("Class Balance:", target_analysis["class_balance"])
                st.write("Imbalance Ratio:", target_analysis["imbalance_ratio"])
                st.write("Is Imbalanced:", target_analysis["is_imbalanced"])

            with st.expander("Quick Dataset Profile"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dataset Size", dataset_profile["size_label"])
                col_b.metric("Total Missing Cells", dataset_profile["total_missing"])
                col_c.metric("Missing %", dataset_profile["missing_percentage"])

        # ===============================
        # DATASET STRUCTURE
        # ===============================
        elif section == "Dataset Structure":
            st.header("Dataset Structure")

            with st.expander("Dataset Preview", expanded=True):
                st.dataframe(df.head())

            with st.expander("Column Names"):
                st.write(column_names)

            with st.expander("Data Types"):
                st.dataframe(data_types)

            with st.expander("Null Values"):
                st.dataframe(null_info)

            with st.expander("Detected Column Types", expanded=True):
                st.write("Numerical Columns:", column_types["numerical"])
                st.write("Categorical Columns:", column_types["categorical"])
                st.write("Text Columns:", column_types["text"])
                st.write("Datetime Columns:", column_types["datetime"])
                st.write("Boolean Columns:", column_types["boolean"])
                st.write("ID-like Columns:", column_types["id_like"])

            with st.expander("Advanced Dataset Profile", expanded=True):
                st.write("Numerical Column Count:", dataset_profile["numerical_count"])
                st.write("Categorical Column Count:", dataset_profile["categorical_count"])
                st.write("Text Column Count:", dataset_profile["text_count"])
                st.write("Datetime Column Count:", dataset_profile["datetime_count"])
                st.write("Boolean Column Count:", dataset_profile["boolean_count"])
                st.write("ID-like Column Count:", dataset_profile["id_like_count"])
                st.write("High Cardinality Categorical Columns:", dataset_profile["high_cardinality_cols"])

        # ===============================
        # ML INSIGHTS
        # ===============================
        elif section == "ML Insights":
            st.header("Machine Learning Insights")

            with st.expander("Detected ML Problem Type", expanded=True):
                st.write("Problem Type:", task_info["problem_type"])
                st.write("Possible Target Column:", task_info["possible_target"])
                st.write("Reason:", task_info["reason"])

            with st.expander("Target Analysis"):
                st.write("Target Found:", target_analysis["target_found"])
                st.write("Unique Values in Target:", target_analysis["unique_values"])
                st.write("Class Balance:", target_analysis["class_balance"])
                st.write("Imbalance Ratio:", target_analysis["imbalance_ratio"])
                st.write("Is Imbalanced:", target_analysis["is_imbalanced"])

            with st.expander("Recommended Preprocessing Steps", expanded=True):
                for step in preprocessing_steps:
                    st.write(f"• {step}")

            with st.expander("Feature Insights"):
                st.write("Top Correlated Numerical Features:", feature_insights["top_correlated_features"])
                st.write("Low Information Features:", feature_insights["low_information_features"])
                st.write("Drop Candidate Features:", feature_insights["drop_candidate_features"])
                st.write("Notes:")
                for note in feature_insights["notes"]:
                    st.write(f"• {note}")

            with st.expander("Dataset Risks / Warnings"):
                for risk in detected_risks:
                    st.write(f"• {risk}")

            with st.expander("Recommended ML Models", expanded=True):
                for model, reason in zip(models, reasons):
                    st.write(f"**{model}**")
                    st.caption(reason)

        # ===============================
        # VISUALIZATIONS
        # ===============================
        elif section == "EDA Visualizations":
            st.header("Automatic EDA Visualizations")

            if missing_fig is not None:
                st.pyplot(missing_fig)
            else:
                st.info("No missing value chart available.")

            if target_fig is not None:
                st.pyplot(target_fig)
            else:
                st.info("No target distribution chart available for this dataset.")

            if corr_fig is not None:
                st.pyplot(corr_fig)
            else:
                st.info("No correlation heatmap available for this dataset.")

        # ===============================
        # CHATBOT
        # ===============================
        elif section == "Chatbot":
            st.header("Dataset Chatbot")

            user_query = st.text_input(
                "Ask something about the dataset",
                placeholder="Example: Is this dataset imbalanced?"
            )

            if user_query:
                bot_response = answer_user_query(
                    user_query=user_query,
                    basic_info=basic_info,
                    dataset_profile=dataset_profile,
                    column_types=column_types,
                    task_info=task_info,
                    target_analysis=target_analysis,
                    preprocessing_steps=preprocessing_steps,
                    feature_insights=feature_insights,
                    detected_risks=detected_risks,
                    models=models,
                    summary_text=summary_text,
                    null_info=null_info
                )

                st.markdown("**Assistant Response:**")
                st.write(bot_response)

    except Exception as e:
        st.error(f"Error while reading file: {e}")

else:
    st.info("Please upload a dataset to continue.")