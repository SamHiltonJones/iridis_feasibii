import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    data = []
    previous_step = 0
    cumulative_step = 0
    reset_occurred = False

    with open(file_path, 'r') as file:
        for line in file:
            if 'Step:' in line and 'Reward:' in line:
                parts = line.strip().split(', ')
                step = int(parts[0].split(': ')[1])
                reward = float(parts[1].split(': ')[1])
                
                if step <= previous_step:
                    reset_occurred = True
                    cumulative_step += previous_step
                previous_step = step
                
                if reset_occurred:
                    reward *= 15

                data.append((cumulative_step + step, reward))

    return pd.DataFrame(data, columns=['Step', 'Reward'])

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

log_file_path = 'reward_log_simple_model_scene2.txt'

df = parse_log_file(log_file_path)

window_size = 50 
df['Moving_Avg_Reward'] = moving_average(df['Reward'], window_size)

plt.figure(figsize=(12, 6))
plt.plot(df['Step'], df['Moving_Avg_Reward'], label='Moving Average Reward')
plt.xlabel('Timesteps')
plt.ylabel('Moving Average Reward')
plt.title('Moving Average of Reward vs Timesteps')
plt.legend()
plt.grid(True)
plt.show()
