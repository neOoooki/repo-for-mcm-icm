import pandas as pd
import numpy as np
import re
import os

def load_and_process_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # List to store processed records
    processed_records = []
    
    # Iterate through each contestant
    for _, row in df.iterrows():
        season = row['season']
        celebrity = row['celebrity_name']
        partner = row['ballroom_partner']
        age = row['celebrity_age_during_season']
        industry = row['celebrity_industry']
        result_str = str(row['results'])
        placement = row['placement']
        
        # Determine elimination week
        elimination_week = None
        if "Eliminated Week" in result_str:
            try:
                elimination_week = int(re.search(r"Eliminated Week (\d+)", result_str).group(1))
            except:
                pass
        elif "Place" in result_str:
            elimination_week = 99 # Never eliminated in standard rounds
        elif "Withdrew" in result_str:
             if "Week" in result_str:
                 try:
                    elimination_week = int(re.search(r"Week (\d+)", result_str).group(1))
                 except:
                    elimination_week = 99
        
        # Iterate through weeks 1 to 15 (covering all possible weeks)
        max_weeks = 15 
        
        for week in range(1, max_weeks + 1):
            # Check if scores exist for this week
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            
            # Check which columns actually exist in df
            existing_cols = [c for c in judge_cols if c in df.columns]
            
            if not existing_cols:
                continue
                
            scores = []
            num_judges = 0
            
            for col in existing_cols:
                val = row[col]
                # Handle string numbers and N/A
                if pd.notna(val) and str(val).strip().upper() != 'N/A' and str(val).strip() != '':
                    try:
                        scores.append(float(val))
                        num_judges += 1
                    except ValueError:
                        pass
            
            # If no scores, assume did not compete this week
            if not scores:
                continue
                
            total_score = sum(scores)
            
            is_eliminated = 0
            if elimination_week == week:
                is_eliminated = 1
                
            record = {
                'season': season,
                'week': week,
                'celebrity_name': celebrity,
                'partner': partner,
                'age': age,
                'industry': industry,
                'total_judge_score': total_score,
                'num_judges': num_judges,
                'is_eliminated': is_eliminated,
                'placement': placement,
                'result_description': result_str
            }
            processed_records.append(record)
            
    # Create DataFrame
    processed_df = pd.DataFrame(processed_records)
    
    if processed_df.empty:
        print("No records processed.")
        return

    # Sort
    processed_df.sort_values(by=['season', 'week', 'total_judge_score'], ascending=[True, True, False], inplace=True)
    
    # Save
    print(f"Saving processed data to {output_path}...")
    processed_df.to_csv(output_path, index=False)
    print(f"Done. Processed {len(processed_df)} records.")
    print("Sample data:")
    print(processed_df.head())

if __name__ == "__main__":
    # Use absolute paths or relative to cwd
    input_csv = os.path.join("data", "raw_data.csv")
    output_csv = os.path.join("data", "processed_data.csv")
    
    load_and_process_data(input_csv, output_csv)
