import os
import glob
import pandas as pd

def find_csv_files(assets_dir="assets"):
    """
    Search for CSV files in the specified assets directory, excluding output files.
    """
    if not os.path.exists(assets_dir):
        if os.path.exists("assest"):
            assets_dir = "assest"
        else:
            raise FileNotFoundError(f"Assets directory '{assets_dir}' not found.")
            
    pattern = os.path.join(assets_dir, "*.csv")
    csv_files = glob.glob(pattern)
    
    # Exclude output combined file
    valid_files = [
        f for f in csv_files 
        if os.path.basename(f).lower() not in ["combined_students.csv"]
    ]
    return assets_dir, sorted(valid_files)

def process_students_performance(df):
    """
    Extract math, reading, and writing scores from StudentsPerformance.csv
    and calculate average percentage per student.
    """
    score_cols = [c for c in df.columns if any(s in c.lower() for s in ['math score', 'reading score', 'writing score'])]
    if score_cols:
        return df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    return pd.Series([0.0] * len(df))

def process_student_marks(df):
    """
    Extract total marks/percentage from Student_Marks.csv.
    """
    if 'Marks' in df.columns:
        return pd.to_numeric(df['Marks'], errors='coerce')
    elif 'marks' in df.columns:
        return pd.to_numeric(df['marks'], errors='coerce')
    else:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            return pd.to_numeric(df[numeric_cols[0]], errors='coerce')
        return pd.Series([0.0] * len(df))

def process_generic_dataset(df):
    """
    Extract percentage/marks for any additional dataset present in assets/.
    """
    col_map = {c.lower(): c for c in df.columns}
    
    # Direct percentage/marks column match
    for key in ['overall_percentage', 'percentage', 'total_marks', 'marks', 'score']:
        if key in col_map:
            return pd.to_numeric(df[col_map[key]], errors='coerce')
            
    # Subject mark/score columns average
    score_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['mark', 'score', 'math', 'science', 'english', 'studies', 'language'])]
    if score_cols:
        return df[score_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        
    # Numeric columns fallback
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        
    return pd.Series([0.0] * len(df))

def extract_student_name(df):
    """
    Extract existing student name column if available.
    """
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for key in ['student_name', 'name', 'full_name', 'studentname']:
        if key in col_map:
            return df[col_map[key]]
    return None

def extract_student_id(df):
    """
    Extract existing student ID column if available.
    """
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for key in ['student_id', 'id', 'roll_no', 'roll_number', 'registration_no']:
        if key in col_map:
            return df[col_map[key]]
    return None

def load_and_normalize_data(assets_dir="assets"):
    """
    Load all student CSV files from assets/, clean and normalize into a unified format:
    student_id, student_name, overall_percentage
    """
    actual_dir, csv_files = find_csv_files(assets_dir)
    print(f"Found {len(csv_files)} CSV file(s) in '{actual_dir}':")
    for f in csv_files:
        print(f" - {os.path.basename(f)}")

    all_records = []
    global_student_counter = 1

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing {filename}...")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if df.empty:
            print(f"Skipping empty file: {filename}")
            continue

        # Extract overall percentage based on file schema
        filename_lower = filename.lower()
        if "studentsperformance" in filename_lower:
            percentages = process_students_performance(df)
        elif "student_marks" in filename_lower and "simple" not in filename_lower:
            percentages = process_student_marks(df)
        else:
            percentages = process_generic_dataset(df)

        # Extract or generate names
        names = extract_student_name(df)
        
        # Extract or generate IDs
        ids = extract_student_id(df)

        # Build normalized records for this DataFrame
        normalized_df = pd.DataFrame()

        num_rows = len(df)
        
        # 1. Student ID
        if ids is not None and not ids.isna().all():
            normalized_df['student_id'] = ids.astype(str)
            # Fill missing IDs if any
            mask_no_id = normalized_df['student_id'].isna() | (normalized_df['student_id'] == 'nan') | (normalized_df['student_id'] == '')
            for i in range(num_rows):
                if mask_no_id.iloc[i]:
                    normalized_df.iloc[i, normalized_df.columns.get_loc('student_id')] = f"STU{global_student_counter:05d}"
                    global_student_counter += 1
                else:
                    global_student_counter += 1
        else:
            generated_ids = [f"STU{global_student_counter + i:05d}" for i in range(num_rows)]
            global_student_counter += num_rows
            normalized_df['student_id'] = generated_ids

        # 2. Student Name
        if names is not None:
            raw_names = names.astype(str)
            cleaned_names = []
            for i, name in enumerate(raw_names):
                if pd.isna(name) or name.strip().lower() in ['nan', 'none', '']:
                    cleaned_names.append(f"Student_{normalized_df['student_id'].iloc[i]}")
                else:
                    cleaned_names.append(name.strip())
            normalized_df['student_name'] = cleaned_names
        else:
            normalized_df['student_name'] = [f"Student_{sid}" for sid in normalized_df['student_id']]

        # 3. Overall Percentage
        normalized_df['overall_percentage'] = pd.to_numeric(percentages, errors='coerce').round(2)

        all_records.append(normalized_df)
        print(f"Extracted {len(normalized_df)} records from {filename}.")

    if not all_records:
        print("No student records processed.")
        return pd.DataFrame(columns=['student_id', 'student_name', 'overall_percentage'])

    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df

def main():
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)
    
    combined_df = load_and_normalize_data(assets_dir=assets_dir)
    
    output_file = os.path.join(assets_dir, "combined_students.csv")
    combined_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 50)
    print(f"Successfully combined {len(combined_df)} records into '{output_file}'.")
    print("=" * 50)
    print("\nSample records (First 10):")
    print(combined_df.head(10).to_string(index=False))
    print("\nSample records (Last 10):")
    print(combined_df.tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
