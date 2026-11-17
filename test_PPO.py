
from pathlib import Path
from stable_baselines3 import PPO
from RC_gymnasium_random import RC_Env

# 1. 建立環境
env = RC_Env()

# 2. 讀取已訓練的 PPO 模型
folder = Path(__file__).resolve().parent
model = PPO.load(str(folder / "RC_PPO_model.zip"))

# 3. 指定測試電阻
test_resistors = [1500, 1800, 2000, 2400, 2700]

success_count = 0

# 4. 開始測試
for initial_r in test_resistors:

    state, info = env.reset()

    # 指定這回合的初始電阻
    env.resistor = initial_r
    env.steps = 0
    state = env.get_state()

    print("======================")
    print("初始電阻:", initial_r, "ohm")

    # 已符合目標的初始狀態也視為成功
    success = abs(
        env.calculate_vout() - env.target_v
    ) <= 0.1

    # 5. 讓 PPO 自己選擇 Action
    while not success:

        action, _ = model.predict(
            state,
            deterministic=True
        )

        state, reward, terminated, truncated, info = env.step(
            int(action)
        )

        print("Action:", int(action))
        print("目前 R1:", env.resistor)
        print("Vout:", round(info["vout"], 3))
        print("Reward:", round(reward, 3))
        print("----------------------")

        success = terminated

        if terminated or truncated:
            break

    # 6. 統計結果
    if success:
        success_count += 1
        print("結果：成功")
    else:
        print("結果：失敗")

    print("最後 R1:", env.resistor)
    print("使用步數:", env.steps)

# 7. 計算成功率
success_rate = success_count / len(test_resistors) * 100

print("======================")
print("成功次數:", success_count)
print("測試次數:", len(test_resistors))
print(f"成功率: {success_rate:.1f}%")

env.close()
