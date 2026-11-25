
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from PyLTSpice import SimRunner, SpiceEditor, LTspice, RawRead


class RC2ParameterEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # 1. 電路目標
        self.target_voltage = 3.0
        self.target_time = 0.020
        self.max_steps = 20

        # 2. R、C 範圍
        self.min_r = 1000
        self.max_r = 5000
        self.min_c = 5
        self.max_c = 20

        # 3. Action Space
        self.action_space = spaces.Discrete(5)

        # 4. Observation Space
        # State = [R/5000, C/20, Vout/5]
        self.observation_space = spaces.Box(
            low=np.array([0.2, 0.25, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 5. 初始參數
        self.resistor = 2000
        self.capacitance_uf = 10
        self.steps = 0

        # 6. LTspice 設定
        folder = Path(__file__).resolve().parent
        self.circuit = folder / "RC_test.asc"

        # 填入之前成功執行的 LTspice.exe 路徑
        ltspice_path = r"D:\LTspice.exe"

        my_ltspice = LTspice.create_from(ltspice_path)

        self.runner = SimRunner(
            simulator=my_ltspice,
            output_folder=str(folder / "simulation_results")
        )

        # 7. 快取模擬結果
        self.cache = {}

    # 8. 執行 LTspice 模擬
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
                f"RC_R{self.resistor}_C"
                f"{self.capacitance_uf}u.net"
            )
        )

        if raw_file is None:
            raise RuntimeError("LTspice 模擬失敗")

        raw = RawRead(str(raw_file), traces_to_read=["time", "V(Vout)"])

        time = raw.get_axis()
        vout = raw.get_trace("V(Vout)").get_wave()

        voltage = float(
            np.interp(self.target_time, time, vout)
        )

        self.cache[key] = voltage

        return voltage

    # 9. 取得 State
    def get_state(self):

        voltage = self.calculate_vout()

        state = np.array([
            self.resistor / self.max_r,
            self.capacitance_uf / self.max_c,
            voltage / 5.0
        ], dtype=np.float32)

        return state

    # 10. 重設環境
    
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

    # 隨機產生 R1：1000～5000Ω
    # 每次間隔 200Ω
        self.resistor = int(
        self.np_random.integers(5, 26) * 200
    )

    # 隨機產生 C1：5～20µF
    # 每次間隔 1µF
        self.capacitance_uf = int(
        self.np_random.integers(5, 21)
    )

        self.steps = 0

        state = self.get_state()

    # 確認 State 沒有超出 Observation Space
        assert self.observation_space.contains(state), \
        "State 超出 Observation Space 範圍"

        return state, {}


    # 11. 執行 Action
    def step(self, action):

        action = int(action)
        self.steps += 1

        if action == 0:
            self.resistor -= 200

        elif action == 1:
            self.resistor += 200

        elif action == 2:
            self.capacitance_uf -= 1

        elif action == 3:
            self.capacitance_uf += 1

        elif action == 4:
            pass

        else:
            raise ValueError("無效的 Action")

        self.resistor = int(np.clip(
            self.resistor,
            self.min_r,
            self.max_r
        ))

        self.capacitance_uf = int(np.clip(
            self.capacitance_uf,
            self.min_c,
            self.max_c
        ))

        # 執行模擬
        vout = self.calculate_vout()

        # 第16課先使用最簡單的 Reward
        error = abs(vout - self.target_voltage)
        reward = -error

        terminated = bool(error <= 0.1)
        truncated = bool(self.steps >= self.max_steps)

        state = self.get_state()

        info = {
            "resistor": self.resistor,
            "capacitance_uf": self.capacitance_uf,
            "vout": vout,
            "error": error
        }

        return state, reward, terminated, truncated, info


# 12. 手動測試
if __name__ == "__main__":

    env = RC2ParameterEnv()

    state, info = env.reset()

    print("初始 R1:", env.resistor)
    print("初始 C1:", env.capacitance_uf)
    print("初始 Vout:", round(env.calculate_vout(), 3))

    actions = [2, 1, 1]

    for action in actions:

        state, reward, terminated, truncated, info = env.step(action)

        print("----------------")
        print("Action:", action)
        print("R1:", info["resistor"])
        print("C1:", info["capacitance_uf"])
        print("Vout:", round(info["vout"], 3))
        print("Reward:", round(reward, 3))
        print("State:", state)

        if terminated or truncated:
            print("Episode 結束")
            break

    env.close()
