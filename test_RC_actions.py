
from RC_2parameter import RC2ParameterEnv

env = RC2ParameterEnv()

# 五種 Action 都從相同狀態開始
for action in range(5):

    state, info = env.reset(seed=42)

    state, reward, terminated, truncated, info = env.step(action)

    print("----------------------")
    print("Action:", action)
    print("R1:", info["resistor"])
    print("C1:", info["capacitance_uf"])
    print("Vout:", round(info["vout"], 3))
    print("Reward:", round(reward, 3))
    print("Terminated:", terminated)

env.close()
