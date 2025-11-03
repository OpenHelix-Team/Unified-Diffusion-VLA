import os
import json
import numpy as np

# corresponding to the evaluation results saved
exp_name = 'univla_calvin_abcd_video'
path = f'/share/project/yuqi.wang/UniVLA/logs/calvin_exp_main/{exp_name}/eval'

def compute_average_scores(path, num_files=8):
    # Initialize accumulators
    total_avg_seq_len = 0
    total_chain_sr = {str(i): 0 for i in range(1, 6)}
    
    for i in range(num_files):
        json_path = os.path.join(path, f'results_calvin_rand-{i}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Accumulate scores
        total_avg_seq_len += data['null']['avg_seq_len']
        
        # Accumulate chain success rates
        for length in range(1, 6):
            total_chain_sr[str(length)] += data['null']['chain_sr'][str(length)]
    
    # Calculate averages
    avg_seq_len = total_avg_seq_len / num_files
    avg_chain_sr = {length: value / num_files for length, value in total_chain_sr.items()}
    
    # Print results
    print(f"Average sequence length across {num_files} runs: {avg_seq_len:.4f}")
    print("Average chain success rates:")
    for length, rate in avg_chain_sr.items():
        print(f"  Length {length}: {rate:.4f}")
    
    return {
        'avg_seq_len': avg_seq_len,
        'chain_sr': avg_chain_sr
    }

results = compute_average_scores(path)