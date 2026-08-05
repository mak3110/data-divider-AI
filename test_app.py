import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import app
from app.segregation import classify_percentage, process_and_segregate_dataframe

client = TestClient(app)

def test_health_check():
    """
    Test that the /health endpoint returns status 200 and healthy status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "Student Segregation & Insights API"

def test_segregation_classification_logic():
    """
    Test that segregation logic correctly classifies specific percentages:
    - 88% -> Group A (Advanced)
    - 75% -> Group B (Intermediate)
    - 62% -> Group C (Foundational Support)
    """
    assert classify_percentage(88.0) == "Group A (Advanced)"
    assert classify_percentage(75.0) == "Group B (Intermediate)"
    assert classify_percentage(62.0) == "Group C (Foundational Support)"

def test_segregate_custom_bracket_mapping():
    """
    Test custom CSV upload endpoint with exact test cases:
    88% (Group A), 75% (Group B), and 62% (Group C).
    """
    csv_content = """student_id,student_name,overall_percentage
STU001,Student One,88.0
STU002,Student Two,75.0
STU003,Student Three,62.0
"""
    files = {
        "file": ("test_students.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }
    response = client.post("/segregate/custom", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_students"] == 3
    assert data["counts"]["Group A (Advanced)"] == 1
    assert data["counts"]["Group B (Intermediate)"] == 1
    assert data["counts"]["Group C (Foundational Support)"] == 1

    group_a_ids = [s["student_id"] for s in data["groups"]["Group A (Advanced)"]]
    group_b_ids = [s["student_id"] for s in data["groups"]["Group B (Intermediate)"]]
    group_c_ids = [s["student_id"] for s in data["groups"]["Group C (Foundational Support)"]]

    assert "STU001" in group_a_ids
    assert "STU002" in group_b_ids
    assert "STU003" in group_c_ids

def test_segregate_dataset_endpoint():
    """
    Test the /segregate/dataset endpoint against the combined student dataset.
    """
    response = client.post("/segregate/dataset")
    assert response.status_code == 200
    data = response.json()
    assert "total_students" in data
    assert "counts" in data
    assert "groups" in data
    assert data["total_students"] > 0
    assert "Group A (Advanced)" in data["counts"]

def test_student_insight_endpoint():
    """
    Test POST /student/insight with an existing student name.
    """
    payload = {"full_name": "Gregory Homenick"}
    response = client.post("/student/insight", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "full_name" in data
    assert "overall_percentage" in data
    assert "group" in data
    assert "academic_insight" in data
    assert isinstance(data["academic_insight"], str)
    assert len(data["academic_insight"]) > 0

def test_student_insight_not_found():
    """
    Test POST /student/insight with a non-existent student name.
    """
    payload = {"full_name": "NonExistentStudent12345"}
    response = client.post("/student/insight", json=payload)
    assert response.status_code == 404
