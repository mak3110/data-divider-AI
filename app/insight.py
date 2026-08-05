import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
from app.segregation import classify_percentage

# Load environment variables from .env and api.env
load_dotenv()
load_dotenv("api.env")

def get_gemini_client():
    """
    Initialize and return the Google GenAI Client using GEMINI_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=api_key)

def find_student(full_name: str, csv_path: str = "assets/combined_students.csv"):
    """
    Find student record matching full_name in combined_students.csv.
    """
    if not os.path.exists(csv_path):
        fallback_path = os.path.join("assest", "combined_students.csv")
        if os.path.exists(fallback_path):
            csv_path = fallback_path
        else:
            raise FileNotFoundError(f"Combined students CSV file not found at '{csv_path}'.")

    df = pd.read_csv(csv_path)
    if df.empty or 'student_name' not in df.columns:
        return None

    query = full_name.strip().lower()
    
    # 1. Exact match (case-insensitive)
    exact_matches = df[df['student_name'].str.strip().str.lower() == query]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
        return {
            "student_id": str(row["student_id"]),
            "full_name": str(row["student_name"]),
            "overall_percentage": float(row["overall_percentage"])
        }

    # 2. Substring match
    substr_matches = df[df['student_name'].str.strip().str.lower().str.contains(query, regex=False)]
    if not substr_matches.empty:
        row = substr_matches.iloc[0]
        return {
            "student_id": str(row["student_id"]),
            "full_name": str(row["student_name"]),
            "overall_percentage": float(row["overall_percentage"])
        }

    return None

def generate_fallback_insight(full_name: str, overall_percentage: float, group_name: str) -> str:
    """
    Generate a structured 3-bullet-point academic report when AI API quota is unavailable.
    """
    if "Group A" in group_name or overall_percentage >= 85.0:
        return (
            f"- Current Performance & Strengths: Demonstrates exceptional subject mastery and consistent academic excellence with an overall score of {overall_percentage}%.\n"
            f"- Recommended Extra Class Focus Areas: Focus on advanced problem-solving techniques, peer mentoring, and participating in honors/competitive academic challenges.\n"
            f"- Motivational Action Plan: Maintain disciplined study habits, explore enrichment topics beyond the core curriculum, and aim for top percentile distinctions."
        )
    elif "Group B" in group_name or overall_percentage >= 70.0:
        return (
            f"- Current Performance & Strengths: Shows solid foundational understanding and dependable core academic performance with an overall score of {overall_percentage}%.\n"
            f"- Recommended Extra Class Focus Areas: Target moderate score subjects to bridge performance gaps and build concept consistency.\n"
            f"- Motivational Action Plan: Establish a targeted weekly revision schedule and practice sample exam papers to elevate overall performance into Group A."
        )
    else:
        return (
            f"- Current Performance & Strengths: Displays foundational potential with an overall score of {overall_percentage}%, with clear opportunities for immediate improvement.\n"
            f"- Recommended Extra Class Focus Areas: Prioritize foundational concepts in core subjects through dedicated remedial and extra assistance classes.\n"
            f"- Motivational Action Plan: Commit to a structured daily study routine, leverage teacher tutoring sessions, and track weekly progress milestones."
        )

def generate_student_insight(full_name: str, overall_percentage: float, group_name: str) -> str:
    """
    Query Google Gemini API (gemini-2.0-flash) to generate a concise 3-bullet-point academic report.
    Falls back to a structured academic insight report if API quota or key is unavailable.
    """
    prompt = f"""You are an expert academic advisor. Generate a concise 3-bullet-point academic report for the following student based on their academic score.

Student Name: {full_name}
Overall Percentage Score: {overall_percentage}%
Performance Bracket: {group_name}

Please format your response strictly as 3 bullet points with the following headings:
- Current Performance & Strengths: <brief analysis based on score>
- Recommended Extra Class Focus Areas: <brief recommendations>
- Motivational Action Plan: <encouraging actionable next steps>

Keep each bullet point concise, constructive, and directly aligned with their score bracket ({group_name})."""

    try:
        client = get_gemini_client()
        # Attempt Gemini 2.0 Flash
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception as e:
        print(f"Gemini API query encountered issue: {e}. Falling back to structured academic insight.")

    # Return structured fallback report if API call fails or quota is exhausted
    return generate_fallback_insight(full_name, overall_percentage, group_name)
