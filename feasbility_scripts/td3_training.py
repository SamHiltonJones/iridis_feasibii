import numpy as np
from stable_baselines3 import TD3
import torch as th
import sys
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from datetime import datetime

def create_env(env_path, worker_id=6, time_scale=1.0, no_graphics = True):
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

    scene1_env = r"scene1_builds/3DPos.exe"
    def make_env_scene1(worker_id):
        return lambda: create_env(scene1_env, worker_id, time_scale=1.0, no_graphics=True)

    env_fns_scene1 = [make_env_scene1(i) for i in range(n_envs)]
    env_scene1 = SubprocVecEnv(env_fns_scene1)

    base_path = 'logs_models'
    model_name = 'td3_model_scene1'
    save_path_scene1 = get_save_path(base_path, model_name)
    checkpoint_callback_scene1 = CheckpointCallback(save_freq=100000, save_path=save_path_scene1, name_prefix=model_name)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[128, 256, 256, 128])

    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)
    model = TD3("MlpPolicy", env_scene1, verbose=2, tensorboard_log=tensorboard_log_path_scene1, policy_kwargs=policy_kwargs, learning_rate=3e-4)
    model.learn(total_timesteps=500000, reset_num_timesteps=True, tb_log_name="train_scene1", callback=[checkpoint_callback_scene1])
    final_model_path_scene1 = get_save_path("trained_models", model_name)
    model.save(final_model_path_scene1)

    env_scene1.close()

    scene2_env = r"scene2_builds/3DPos.exe"
    env_scene2 = create_env(scene2_env, worker_id=7, time_scale=1.0, no_graphics=True)

    model_name = 'td3_model_scene2'
    save_path_scene2 = get_save_path(base_path, model_name)
    checkpoint_callback_scene2 = CheckpointCallback(save_freq=100000, save_path=save_path_scene2, name_prefix=model_name)
    tensorboard_log_path_scene2 = get_save_path("./logs_graphs", model_name)

    model = TD3.load(final_model_path_scene1, env=env_scene2, tensorboard_log=tensorboard_log_path_scene2, policy_kwargs=policy_kwargs, verbose=2, learning_rate=1e-4)
    
    reward_logging_callback_scene2 = RewardLoggingCallback(log_interval=2000, log_file=f'reward_log_{model_name}.txt')

    model.learn(total_timesteps=500000, reset_num_timesteps=False, tb_log_name="finetune_scene2", callback=[checkpoint_callback_scene2, reward_logging_callback_scene2])
    final_model_path_scene2 = get_save_path("trained_models", model_name)
    model.save(final_model_path_scene2)

    env_scene2.close()

    scene3_env = r"scene3_builds/3DPos.exe"
    env_scene3 = create_env(scene3_env, worker_id=9, time_scale=1.0, no_graphics=True)

    model_name = 'td3_model_scene3'
    save_path_scene3 = get_save_path(base_path, model_name)
    checkpoint_callback_scene3 = CheckpointCallback(save_freq=100000, save_path=save_path_scene3, name_prefix=model_name)
    tensorboard_log_path_scene3 = get_save_path("./logs_graphs", model_name)

    model = TD3.load(final_model_path_scene2, env=env_scene3, tensorboard_log=tensorboard_log_path_scene3, policy_kwargs=policy_kwargs, verbose=2, learning_rate=5e-5)

    reward_logging_callback_scene3 = RewardLoggingCallback(log_interval=2000, log_file=f'reward_log_{model_name}.txt')

    model.learn(total_timesteps=500000, reset_num_timesteps=False, tb_log_name="finetune_scene3", callback=[checkpoint_callback_scene3, reward_logging_callback_scene3])
    final_model_path_scene3 = get_save_path("trained_models", model_name)
    model.save(final_model_path_scene3)

    env_scene3.close()
