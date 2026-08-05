import pandas as pd
from typing import Dict, Any, List

def classify_percentage(val: float) -> str:
    """
    Classify a student's overall percentage into standard performance brackets:
    - Group A (Advanced): 85% to 100%
    - Group B (Intermediate): 70% to 84.99%
    - Group C (Foundational Support): Below 70%
    """
    if val >= 85.0:
        return "Group A (Advanced)"
    elif val >= 70.0:
        return "Group B (Intermediate)"
    else:
        return "Group C (Foundational Support)"

def process_and_segregate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a pandas DataFrame of student records, extract or compute normalized
    fields (student_id, student_name, overall_percentage), and segregate students
    into performance brackets.
    """
    if df is None or df.empty:
        return {
            "total_students": 0,
            "counts": {
                "Group A (Advanced)": 0,
                "Group B (Intermediate)": 0,
                "Group C (Foundational Support)": 0
            },
            "groups": {
                "Group A (Advanced)": [],
                "Group B (Intermediate)": [],
                "Group C (Foundational Support)": []
            }
        }

    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}

    # 1. Extract or compute overall_percentage
    if "overall_percentage" in col_map:
        perc_series = pd.to_numeric(df[col_map["overall_percentage"]], errors='coerce').fillna(0.0)
    elif "marks" in col_map:
        perc_series = pd.to_numeric(df[col_map["marks"]], errors='coerce').fillna(0.0)
    elif "percentage" in col_map:
        perc_series = pd.to_numeric(df[col_map["percentage"]], errors='coerce').fillna(0.0)
    else:
        # Average across score/mark columns if specific percentage column is absent
        score_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['score', 'mark', 'math', 'reading', 'writing', 'science', 'english', 'studies', 'language'])]
        if score_cols:
            perc_series = df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0.0)
        else:
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                perc_series = df[num_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0.0)
            else:
                perc_series = pd.Series([0.0] * len(df))

    # 2. Extract or generate student_id
    if "student_id" in col_map:
        id_series = df[col_map["student_id"]].astype(str)
    elif "id" in col_map:
        id_series = df[col_map["id"]].astype(str)
    else:
        id_series = pd.Series([f"STU{i+1:05d}" for i in range(len(df))])

    # 3. Extract or generate student_name
    if "student_name" in col_map:
        name_series = df[col_map["student_name"]].astype(str)
    elif "name" in col_map:
        name_series = df[col_map["name"]].astype(str)
    else:
        name_series = pd.Series([f"Student_{sid}" for sid in id_series])

    groups: Dict[str, List[Dict[str, Any]]] = {
        "Group A (Advanced)": [],
        "Group B (Intermediate)": [],
        "Group C (Foundational Support)": []
    }

    for sid, sname, perc in zip(id_series, name_series, perc_series):
        perc_val = round(float(perc), 2)
        group_name = classify_percentage(perc_val)

        # Generate clean name placeholder if missing/NaN
        sname_str = str(sname).strip()
        if pd.isna(sname) or sname_str.lower() in ['nan', 'none', '']:
            sname_str = f"Student_{sid}"

        student_obj = {
            "student_id": str(sid).strip(),
            "student_name": sname_str,
            "overall_percentage": perc_val
        }
        groups[group_name].append(student_obj)

    counts = {group: len(students) for group, students in groups.items()}

    return {
        "total_students": len(df),
        "counts": counts,
        "groups": groups
    }
