
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from RC_gymnasium_random import RC_Env

# 1. 建立 RC 環境
env = RC_Env()

# 2. 檢查環境
check_env(env)

# 3. 建立 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    n_steps=128,
    batch_size=64
)

# 4. 開始訓練
model.learn(total_timesteps=30000)

# 5. 儲存訓練完成的模型
model.save("RC_PPO_model")
print("PPO 模型已儲存！")

env.close()
