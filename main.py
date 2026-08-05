import os
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.segregation import process_and_segregate_dataframe, classify_percentage
from app.insight import find_student, generate_student_insight

app = FastAPI(
    title="Student Segregation & Insights API",
    description="FastAPI application to segregate students into performance brackets and generate personalized academic insights using Google Gemini API.",
    version="1.1.0"
)

COMBINED_DATASET_PATH = os.path.join("assets", "combined_students.csv")

class StudentInsightRequest(BaseModel):
    full_name: str = Field(..., description="Full name of the student to generate academic insights for", example="Armando Pollich PhD")

@app.get("/health", summary="Health Check")
def health_check():
    """
    Basic API health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "Student Segregation & Insights API",
        "version": "1.1.0"
    }

@app.post("/segregate/dataset", summary="Segregate Combined Dataset")
def segregate_dataset():
    """
    Reads the combined student dataset from assets/combined_students.csv
    and returns a JSON payload grouping students into Group A, Group B, and Group C brackets.
    """
    path_to_read = COMBINED_DATASET_PATH
    if not os.path.exists(path_to_read):
        fallback_path = os.path.join("assest", "combined_students.csv")
        if os.path.exists(fallback_path):
            path_to_read = fallback_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Combined student dataset file not found at '{COMBINED_DATASET_PATH}'. Please run data_loader.py first."
            )

    try:
        df = pd.read_csv(path_to_read)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading combined dataset file: {str(e)}"
        )

    result = process_and_segregate_dataframe(df)
    return JSONResponse(content=result)

@app.post("/segregate/custom", summary="Segregate Custom Uploaded CSV")
async def segregate_custom(file: UploadFile = File(...)):
    """
    Accepts a custom CSV file upload, processes the student records,
    and returns the segregated student breakdown by performance brackets.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid .csv file."
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        
        buffer = io.BytesIO(contents)
        df = pd.read_csv(buffer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse uploaded CSV file: {str(e)}"
        )

    result = process_and_segregate_dataframe(df)
    return JSONResponse(content=result)

@app.post("/student/insight", summary="Generate Personalized Academic Insight")
def get_student_insight(payload: StudentInsightRequest):
    """
    Searches assets/combined_students.csv for the student by full_name,
    retrieves their overall percentage and performance bracket,
    and queries Gemini API to generate a personalized 3-bullet-point academic report.
    """
    try:
        student_record = find_student(payload.full_name, COMBINED_DATASET_PATH)
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf_err)
        )

    if not student_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student '{payload.full_name}' not found in dataset."
        )

    score = student_record["overall_percentage"]
    group = classify_percentage(score)
    full_name = student_record["full_name"]

    try:
        academic_insight = generate_student_insight(full_name, score, group)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate academic insight via Gemini API: {str(err)}"
        )

    return {
        "full_name": full_name,
        "overall_percentage": score,
        "group": group,
        "academic_insight": academic_insight
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
