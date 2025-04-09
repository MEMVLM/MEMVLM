import pandas as pd    
import numpy as np    
from sklearn.metrics import roc_auc_score    
import math    

# Read CSV file    
df = pd.read_csv('eval/mistralai_4.8_eval_cogvlm1.csv')    

# PSNR calculation function    
def psnr(mse):    
    return 10 * math.log10(1 / mse) if mse > 0 else float('-inf')  # Add check for mse == 0 to avoid math error    

# Anomaly score calculation function (using min-max normalization)    
def anomaly_score(psnr_list, max_psnr, min_psnr):    
    return 1 - ((psnr_list - min_psnr) / (max_psnr - min_psnr))    

# Modified smoothing function
def anomaly_diffusion_v1(memloss, window_size):
    memloss_new = memloss.copy()
    for i in range(window_size, len(memloss) - window_size):
        # Replace the current item with the maximum value within the window range
        memloss_new[i] = max(memloss[i - window_size:i + window_size + 1])
    return memloss_new

def anomaly_diffusion_v2(memloss, window_size, gama = 0.3):
    memloss_new = memloss.copy()
    for i in range(window_size, len(memloss) - window_size):
        window = memloss[i - window_size:i + window_size + 1]
        non_zero_count = np.count_nonzero(window != 0)  # 统计窗口内 0 的个数
        threshold = int(len(window)*gama)
        if non_zero_count >= threshold:
            memloss_new[i] =  max(memloss[i - window_size:i + window_size + 1])  # 如果超过阈值，则当前帧设为 0
    
    return memloss_new


# Initialize lists for overall AUC comparison    
all_anomaly_scores_psnr = []    
all_anomaly_scores_scores = []    
all_anomaly_scores_combined = []    
all_labels = []    

# Dictionary to store AUC per scene    
scene_auc_psnr = {}    
scene_auc_scores = {}    
scene_auc_combined = {}    

# Smoothing window size (number of frames to look before and after the current frame)    
smoothing_window = 6


# Group by video_id (extracted from the path) and process each group    
for video_id, group in df.groupby(df['image_path'].str.extract(r'/(\d{2}_\d{4})/')[0]):    
#for video_id, group in df.groupby(df['image_path'].str.extract(r'/(\d{2})/')[0]):
    # Extract memory loss, scores, and labels    
    memloss = group['memloss'].values    
    labels = group['label'].values    
    scores = group['score'].values  

    #print(scores)
    # Apply smoothing    
    memloss = anomaly_diffusion_v2(memloss, smoothing_window)    
    scores = anomaly_diffusion_v2(scores, smoothing_window)
    #memloss, scores = smooth_memloss_v5(memloss, scores, smoothing_window)

    # First-time EMA to smooth the preds with a more sensitive way
    memloss = pd.Series(memloss).ewm(alpha = 0.33, adjust=True).mean().to_list()
    scores = pd.Series(scores).ewm(alpha = 0.33, adjust=True).mean().to_list()

    # Calculate PSNR values    
    psnr_values_smooth = [psnr(m) for m in memloss]    

    # Calculate anomaly scores (PSNR-based)
    max_psnr_smooth = max(psnr_values_smooth)
    min_psnr_smooth = min(psnr_values_smooth)
    anomaly_scores_psnr = anomaly_score(np.array(psnr_values_smooth), max_psnr_smooth, min_psnr_smooth)


    # Calculate combined anomaly scores (weighted sum of psnr and scores)
    combined_scores =  anomaly_scores_psnr.copy()
    for i in range (len(anomaly_scores_psnr)):
       combined_scores[i] = anomaly_scores_psnr[i] + 0.1* scores[i]

    # Update overall lists
    all_anomaly_scores_psnr.extend(anomaly_scores_psnr)
    all_anomaly_scores_scores.extend(scores)
    all_anomaly_scores_combined.extend(combined_scores)
    all_labels.extend(labels)

    # Extract scene ID (first two digits of the video_id)
    scene_id = video_id.split('_')[0]

    # Store scores and labels per scene for PSNR-based AUC
    if scene_id not in scene_auc_psnr:
        scene_auc_psnr[scene_id] = {'scores': [], 'labels': []} 
    scene_auc_psnr[scene_id]['scores'].extend(anomaly_scores_psnr)    
    scene_auc_psnr[scene_id]['labels'].extend(labels)

    # Store scores and labels per scene for scores-based AUC
    if scene_id not in scene_auc_scores:
        scene_auc_scores[scene_id] = {'scores': [], 'labels': []}
    scene_auc_scores[scene_id]['scores'].extend(scores)    
    scene_auc_scores[scene_id]['labels'].extend(labels)

    # Store scores and labels per scene for combined AUC
    if scene_id not in scene_auc_combined:
        scene_auc_combined[scene_id] = {'scores': [], 'labels': []}
    scene_auc_combined[scene_id]['scores'].extend(combined_scores)
    scene_auc_combined[scene_id]['labels'].extend(labels)

# Calculate overall AUC (PSNR-based)    
auc_psnr = roc_auc_score(all_labels, all_anomaly_scores_psnr)    
print(f'Overall AUC (PSNR-based): {auc_psnr}')    

# Calculate AUC for each scene (PSNR-based)    
for scene_id, data in scene_auc_psnr.items():    
    scene_auc_value_psnr = roc_auc_score(data['labels'], data['scores'])    
    #print(f'Scene {scene_id} AUC (PSNR-based): {scene_auc_value_psnr}')    


# Calculate overall AUC (scores-based)    
auc_scores = roc_auc_score(all_labels, all_anomaly_scores_scores)    
print(f'Overall AUC (Scores-based): {auc_scores}')    

# Calculate AUC for each scene (Scores-based)    
for scene_id, data in scene_auc_scores.items():    
    scene_auc_value_scores = roc_auc_score(data['labels'], data['scores'])    
    #print(f'Scene {scene_id} AUC (Scores-based): {scene_auc_value_scores}')    

# Calculate overall AUC (combined)    
auc_combined = roc_auc_score(all_labels, all_anomaly_scores_combined)    
print(f'Overall AUC (Combined PSNR + Scores): {auc_combined}')    

# Calculate AUC for each scene (combined)    
for scene_id, data in scene_auc_combined.items():    
    scene_auc_value_combined = roc_auc_score(data['labels'], data['scores'])    
    #print(f'Scene {scene_id} AUC (Combined PSNR + Scores): {scene_auc_value_combined}')

