import json
import pandas as pd

# 1. Load the RSH results
with open('runs/preds_rsh_final.json', 'r') as f:
    preds = json.load(f)

# 2. Extract unique StudyIDs (e.g., Anonymized100) and sort alphabetically
# This is how we ensure examID_1 matches the ground truth
uids = list(preds.keys())
study_ids = sorted(list(set([u.replace('_left', '').replace('_right', '') for u in uids])))

# 3. Map to examID_X and create rows
submission_data = []
for i, study in enumerate(study_ids):
    exam_id = f"examID_{i+1}"
    
    # The leaderboard expects 2 rows per ID (Left and Right)
    for side in ['left', 'right']:
        uid = f"{study}_{side}"
        if uid in preds:
            # Your predict.py nested them: preds[uid][side]
            scores = preds[uid][side]
            submission_data.append({
                'ID': exam_id,
                'normal': scores['normal'],
                'benign': scores['benign'],
                'malignant': scores['malignant']
            })

# 4. Save
pd.DataFrame(submission_data).to_csv('predictions_leaderboard.csv', index=False)
print("Done! Upload predictions_leaderboard.csv to the portal.")