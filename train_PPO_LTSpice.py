
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from RC_LTSpice_env import RCLTspiceEnv

# 1. 設定資料夾
folder = Path(__file__).resolve().parent

# 2. 建立 LTspice 環境
env = RCLTspiceEnv()

# 3. 檢查環境
check_env(env)
print("LTspice 環境檢查成功！")

# 4. 建立 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    n_steps=64,
    batch_size=32,
    learning_rate=0.0003
)

# 5. 開始訓練
print("開始 PPO + LTspice 訓練！")

model.learn(total_timesteps=5000)

# 6. 儲存新模型
save_path = folder / "RC_PPO_LTspice_5000"
model.save(str(save_path))

print("模型訓練完成！")
print("模型已儲存：", save_path)

# 7. 關閉環境
env.close()
