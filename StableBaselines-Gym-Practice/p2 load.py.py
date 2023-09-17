import gymnasium as gym
from stable_baselines3 import DQN

import os
env = gym.make('LunarLander-v2',render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()
models_dir="models/DQN1" #change with algorithm
model_path=f"{models_dir}/30000.zip"


logdir="logs"
model=DQN.load(model_path,env=env)

# model = DQN('MlpPolicy', env, verbose=1,tensorboard_log=logdir)

# TIMESTEPS=10000
# for iter in range(30):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN1")
#     model.save(f"{models_dir}/{TIMESTEPS*iter}") #example: models/DQN/10000*iter EVERY 10000 TIMESTEPS THE MODEL WILL BE SAVED

episodes = 10

for ep in range(episodes):
	obs,infor = env.reset()
	done = False
	while not done:
		env.render()
		# obs=obs.reshape(8,)#reshape to 1 sample
		action, _ = model.predict(obs)
		obs, rewards, done, info, _ = env.step(action)

env.close()