
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from RC_LTSpice_env import RCLTspiceEnv

# 1. 載入 PPO 模型
folder = Path(__file__).resolve().parent
model = PPO.load(str(folder / "RC_PPO_LTspice_5000.zip"))

# 2. 設定測試條件
test_resistors = [1500, 1800, 2000, 2400, 2700]
repeats = 10

# 每組初始電阻重複測試 10 次
test_cases = test_resistors * repeats

# 固定 Random Agent 的隨機種子
rng = np.random.default_rng(42)


# 3. 定義 Agent 測試函式
def evaluate(agent_type):

    env = RCLTspiceEnv()

    success_count = 0
    total_steps = 0
    total_reward = 0

    for initial_r in test_cases:

        state, info = env.reset(
            options={"initial_resistor": initial_r}
        )

        episode_reward = 0
        success = (
            abs(env.calculate_vout() - env.target_v)
            <= 0.1
        )

        while not success:

            # PPO 或 Random 選擇 Action
            if agent_type == "PPO":
                action, _ = model.predict(
                    state,
                    deterministic=True
                )
                action = int(action)

            else:
                action = int(rng.integers(0, 3))

            # 執行動作
            state, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += reward
            success = terminated

            if terminated or truncated:
                break

        if success:
            success_count += 1

        total_steps += env.steps
        total_reward += episode_reward

    # 4. 統計結果
    n = len(test_cases)

    results = {
        "success_rate": success_count / n * 100,
        "avg_steps": total_steps / n,
        "avg_reward": total_reward / n,
        "simulations": len(env.cache)
    }

    env.close()
    return results


# 5. 分別測試 PPO 與 Random
ppo_result = evaluate("PPO")
random_result = evaluate("Random")

# 6. 印出比較結果
print("\n===== PPO vs Random =====")

for name, result in [
    ("PPO", ppo_result),
    ("Random", random_result)
]:
    print(f"\n{name} Agent")
    print(f"Success Rate: {result['success_rate']:.1f}%")
    print(f"Average Steps: {result['avg_steps']:.2f}")
    print(f"Average Reward: {result['avg_reward']:.3f}")
    print(f"LTspice Simulations: {result['simulations']}")
