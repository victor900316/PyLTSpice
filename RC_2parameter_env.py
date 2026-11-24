
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PyLTSpice import SimRunner, SpiceEditor, LTspice, RawRead
from stable_baselines3.common.env_checker import check_env

class RC2ParameterEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # 1. 設計規格
        self.target_voltage = 3.0
        self.target_time = 0.020
        self.max_steps = 20

        # 2. 定義 Action Space
        # 0: R-200, 1: R+200
        # 2: C-1u, 3: C+1u, 4: 不變
        self.action_space = spaces.Discrete(5)

        # 3. 定義 Observation Space
        # [R/5000, C(uF)/20, Vout/5]
        self.observation_space = spaces.Box(
            low=np.array([0.2, 0.25, 0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.resistor = 2000
        self.capacitance_uf = 10
        self.steps = 0

        # 4. 設定 LTspice
        folder = Path(__file__).resolve().parent
        self.circuit = folder / "RC_test.asc"

        ltspice_path = r"D:\LTspice.exe"
        my_ltspice = LTspice.create_from(ltspice_path)

        self.runner = SimRunner(
            simulator=my_ltspice,
            output_folder=str(folder / "simulation_results")
        )

        # 5. 儲存模擬結果
        self.cache = {}

    # 6. 使用 LTspice 取得 Vout
    def calculate_vout(self):

        key = (self.resistor, self.capacitance_uf)

        if key in self.cache:
            return self.cache[key]

        netlist = SpiceEditor(str(self.circuit))

        netlist.set_component_value(
            "R1", str(self.resistor)
        )

        netlist.set_component_value(
            "C1", f"{self.capacitance_uf}u"
        )

        raw_file, log_file = self.runner.run_now(
            netlist,
            run_filename=(
                f"RC_R{self.resistor}_"
                f"C{self.capacitance_uf}u.net"
            )
        )

        if raw_file is None:
            raise RuntimeError("LTspice 模擬失敗")

        raw = RawRead(str(raw_file))

        time = raw.get_trace("time").get_wave()
        vout = raw.get_trace("V(vout)").get_wave()

        voltage = float(
            np.interp(self.target_time, time, vout)
        )

        self.cache[key] = voltage
        return voltage

    # 7. 取得 State
    def get_state(self):
        vout = self.calculate_vout()

        return np.array(
            [
                self.resistor / 5000,
                self.capacitance_uf / 20,
                vout / 5
            ],
            dtype=np.float32
        )

    # 8. 重設環境
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) #seed 用來讓隨機結果可以實現

        self.resistor = int(
            self.np_random.integers(5, 26) * 200 #隨機產生初始電阻(15~27)*100
        )
        self.capacitance_uf = int(
            self.np_random.integers(5, 21) #隨機產生初始電阻(15~27)*100
        )
        self.steps = 0

        return self.get_state(), {}

    # 9. 執行 Action
    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError("無效的 Action")

        if action == 0:
            self.resistor -= 200
        elif action == 1:
            self.resistor += 200
        elif action == 2:
            self.capacitance_uf -= 1
        elif action == 3:
            self.capacitance_uf += 1

        self.resistor = int(
            np.clip(self.resistor, 1000, 5000)
        )

        self.capacitance_uf = int(
            np.clip(self.capacitance_uf, 5, 20)
        )

        self.steps += 1

        vout = self.calculate_vout()

       
        # 1. 計算電壓誤差
        error = abs(vout - self.target_voltage)

        # 2. 計算電容成本
        cost_penalty = 0.02 * (self.capacitance_uf / 10)

        # 3. 判斷設計規格
        voltage_ok = error <= 0.1
        capacitor_ok = self.capacitance_uf <= 8

        # 4. 必須同時符合兩個條件才成功
        success = voltage_ok and capacitor_ok

        # 5. 設定成功獎勵
        bonus = 2.0 if success else 0.0

        # 6. 計算 Reward
        reward = -error - cost_penalty + bonus

        # 7. 判斷 Episode 是否結束
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps)

        state = self.get_state()

        info = {
            "resistor": self.resistor,
            "capacitance_uf": self.capacitance_uf,
            "vout": vout,
            "error": error,
            "voltage_ok": voltage_ok,
            "capacitor_ok": capacitor_ok,
            "success": success
        }

        return (
            state,
            float(reward),
            terminated,
            truncated,
            info
        )


# 10. 測試環境
# if __name__ == "__main__":

#     env = RC2ParameterEnv()

#     state, info = env.reset()

#     print("初始 State:", state)

#     # 測試：先減少電容，再增加電阻
#     for action in [2, 1, 1]:

#         state, reward, terminated, truncated, info = env.step(action)

#         print("------------------")
#         print("Action:", action)
#         print("R1:", info["resistor"], "Ω")
#         print("C1:", info["capacitance_uf"], "µF")
#         print("Vout:", round(info["vout"], 3))
#         print("Reward:", round(reward, 3))

#         if terminated or truncated:
#             print("Episode 結束")
#             break

#     env.close()

if __name__ == "__main__":

    env = RC2ParameterEnv()
    
    state, info = env.reset(seed=42)

    print("初始 R1:", env.resistor)
    print("初始 C1:", env.capacitance_uf)

    for action in [2, 1, 1, 2, 1]:

        state, reward, terminated, truncated, info = env.step(action)

        print("----------------")
        print("Action:", action)
        print("R1:", info["resistor"])
        print("C1:", info["capacitance_uf"])
        print("Vout:", round(info["vout"], 3))
        print("Reward:", round(reward, 3))
        print("Voltage OK:", info["voltage_ok"])
        print("Capacitor OK:", info["capacitor_ok"])
        print("Success:", info["success"])

        if terminated or truncated:
            print("Episode 結束")
            break
    env.close()


