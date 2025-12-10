import re
import os
from collections import defaultdict

directory = '/home/bavish/scratch/cmput655/greedyac_ga/pendulum_v1'
files = [f for f in os.listdir(directory) if '-err' not in f and os.path.isfile(os.path.join(directory, f))]

setting_rewards = defaultdict(list)
setting_configs = defaultdict(str)

for filename in files:
    with open(os.path.join(directory, filename), 'r') as f:
        text = f.read()
    
    setting_match = re.search(r'SETTING_NUM: (\d+)', text)
    if not setting_match:
        continue
    setting_num = int(setting_match.group(1))

    setting_config = re.search(r'Agent setting: (.+)', text)
    
    eval_matches = re.findall(r'=== EVAL ep: \d+, r: ([-\d.]+),', text)
    if len(eval_matches) >= 10:
        last_10 = [float(r) for r in eval_matches[-10:]]
        avg = sum(last_10) / len(last_10)
        setting_rewards[setting_num].append(avg)
        setting_configs[setting_num] = setting_config.group(1) if setting_config else ""

setting_averages = {}
for setting, rewards in setting_rewards.items():
    setting_averages[setting] = sum(rewards) / len(rewards)

for setting in sorted(setting_averages.keys()):
    print(f"Setting {setting}: {setting_averages[setting]:.4f}")

best_setting = max(setting_averages, key=setting_averages.get)
print(f"\nBest setting: {best_setting} with average: {setting_averages[best_setting]:.4f}")
print(f"Configuration: {setting_configs[best_setting]}")