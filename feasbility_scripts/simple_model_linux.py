import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

def create_env(env_path, worker_id=0, time_scale=5.0, no_graphics=True):
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(env_path, side_channels=[channel], worker_id=worker_id, no_graphics=no_graphics)
    channel.set_configuration_parameters(time_scale=time_scale)
    return UnityToGymWrapper(unity_env)

def get_save_path(base_path, model_name, trained_path=False):
    date_str = datetime.now().strftime("%d%m%Y")
    version = 0
    if trained_path:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
    else:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
    return save_path[:-4] if trained_path else save_path

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_interval, log_file, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.log_file = log_file
        self.step_count = 0
        self.total_reward = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_reward += np.sum(self.locals['rewards'])

        if self.step_count % self.log_interval == 0:
            with open(self.log_file, 'a') as f:
                f.write(f'Step: {self.step_count}, Reward: {self.total_reward}\n')
            print(f'Logged reward at step {self.step_count}: {self.total_reward}')
            self.total_reward = 0
        return True

if __name__ == '__main__':
    n_envs = 1

    scene1_env = r"scene2_linux\scene2_linux.x86_64"
    def make_env_scene1(worker_id):
        return lambda: create_env(scene1_env, worker_id, time_scale=5.0, no_graphics=True)

    env_fns_scene1 = [make_env_scene1(i) for i in range(n_envs)]
    env_scene1 = SubprocVecEnv(env_fns_scene1)

    base_path = 'logs_models'
    trained_models_path = 'trained_models'
    os.makedirs(trained_models_path, exist_ok=True)

    model_name = 'simple_model_scene2'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=200000, save_path=save_path_scene1, name_prefix=model_name)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[64, 64], vf=[64, 64])])

    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    reward_logging_callback_scene1 = RewardLoggingCallback(log_interval=1000, log_file=f'reward_log_{model_name}.txt')

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PPO("MlpPolicy", env_scene1, verbose=2, tensorboard_log=tensorboard_log_path_scene1,
                policy_kwargs=policy_kwargs, learning_rate=1e-4, batch_size=256, device=device)

    model.learn(total_timesteps=1000000, reset_num_timesteps=True, tb_log_name="train_scene2",
                callback=[checkpoint_callback_scene1, reward_logging_callback_scene1])

    final_model_path_scene1 = get_save_path(trained_models_path, model_name, trained_path=True)
    model.save(final_model_path_scene1)
    env_scene1.close()
