from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import traceback
import os

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
from visualizer import (
    plot_missing_values,
    plot_target_distribution,
    plot_correlation_heatmap,
    fig_to_base64
)
from chatbot import answer_user_query

app = FastAPI(title="Smart Dataset Assistant API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory data store for local session
STORED_SESSION = {
    "df": None,
    "basic_info": None,
    "dataset_profile": None,
    "column_types": None,
    "task_info": None,
    "target_analysis": None,
    "preprocessing_steps": None,
    "feature_insights": None,
    "detected_risks": None,
    "models": None,
    "summary_text": None,
    "null_info": None
}

class ChatRequest(BaseModel):
    query: str

def sanitize_for_json(obj):
    """Recursively convert numpy / pandas scalars to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is np.nan or obj != obj:  # NaN check
        return None
    else:
        return obj

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Read file contents into memory
        contents = await file.read()
        file_like = io.BytesIO(contents)
        
        # Load dataframe
        df = load_data(file_like, filename=file.filename)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

        # Run analysis pipeline
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

        # Generate Matplotlib plots and convert to base64
        missing_fig = plot_missing_values(df)
        target_fig = plot_target_distribution(df, task_info["possible_target"])
        corr_fig = plot_correlation_heatmap(df, column_types)

        charts = {
            "missing_values": fig_to_base64(missing_fig),
            "target_distribution": fig_to_base64(target_fig),
            "correlation_heatmap": fig_to_base64(corr_fig)
        }

        # Store in session for chat
        STORED_SESSION["df"] = df
        STORED_SESSION["basic_info"] = basic_info
        STORED_SESSION["dataset_profile"] = dataset_profile
        STORED_SESSION["column_types"] = column_types
        STORED_SESSION["task_info"] = task_info
        STORED_SESSION["target_analysis"] = target_analysis
        STORED_SESSION["preprocessing_steps"] = preprocessing_steps
        STORED_SESSION["feature_insights"] = feature_insights
        STORED_SESSION["detected_risks"] = detected_risks
        STORED_SESSION["models"] = models
        STORED_SESSION["summary_text"] = summary_text
        STORED_SESSION["null_info"] = null_info  # Storing original dataframe for chatbot script

        # Prep serialized formats for response
        serialized_data_types = data_types.to_dict(orient="records")
        serialized_null_info = null_info.to_dict(orient="records")

        return sanitize_for_json({
            "success": True,
            "basic_info": basic_info,
            "column_names": column_names,
            "data_types": serialized_data_types,
            "null_info": serialized_null_info,
            "column_types": column_types,
            "dataset_profile": dataset_profile,
            "task_info": task_info,
            "target_analysis": target_analysis,
            "models": models,
            "reasons": reasons,
            "summary_text": summary_text,
            "preprocessing_steps": preprocessing_steps,
            "feature_insights": feature_insights,
            "detected_risks": detected_risks,
            "charts": charts
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze dataset: {str(e)}")

@app.post("/api/chat")
def chat_with_dataset(request: ChatRequest):
    if STORED_SESSION["df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No active dataset session. Please upload a dataset first."
        )

    try:
        response_text = answer_user_query(
            user_query=request.query,
            basic_info=STORED_SESSION["basic_info"],
            dataset_profile=STORED_SESSION["dataset_profile"],
            column_types=STORED_SESSION["column_types"],
            task_info=STORED_SESSION["task_info"],
            target_analysis=STORED_SESSION["target_analysis"],
            preprocessing_steps=STORED_SESSION["preprocessing_steps"],
            feature_insights=STORED_SESSION["feature_insights"],
            detected_risks=STORED_SESSION["detected_risks"],
            models=STORED_SESSION["models"],
            summary_text=STORED_SESSION["summary_text"],
            null_info=STORED_SESSION["null_info"]
        )
        return {"response": response_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Get path of frontend folder relative to backend
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.abspath(os.path.join(current_dir, "../frontend"))

# Mount frontend files under /
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
