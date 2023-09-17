import gymnasium as gym
from stable_baselines3 import DQN

import os

models_dir="models/DQN" #change with algorithm
logdir="logs"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)


if not os.path.exists(logdir):
	os.makedirs(logdir)
	
env = gym.make('LunarLander-v2',render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

model = DQN('MlpPolicy', env, verbose=1,tensorboard_log=logdir)

TIMESTEPS=10000
for iter in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*iter}") #example: models/DQN/10000*iter EVERY 10000 TIMESTEPS THE MODEL WILL BE SAVED

	

# episodes = 10

# for ep in range(episodes):
# 	obs,infor = env.reset()
# 	print(obs)
# 	done = False
# 	while not done:
# 		# obs=obs.reshape(8,)#reshape to 1 sample
# 		action, _ = model.predict(obs)
# 		obs, rewards, done, info, _ = env.step(action)
# 		env.render()
# 		print(rewards)
# env.close()