
from RC_2parameter import RC2ParameterEnv

env = RC2ParameterEnv()

for i in range(5):

    state, info = env.reset()

    print("--------------------")
    print("Episode:", i + 1)
    print("R1:", env.resistor)
    print("C1:", env.capacitance_uf)
    print("Vout:", round(env.calculate_vout(), 3))
    print("State:", state)

    print(
        "State 合法:",
        env.observation_space.contains(state)
    )

env.close()
