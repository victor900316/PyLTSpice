
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PyLTSpice import SimRunner, SpiceEditor, LTspice, RawRead
from stable_baselines3.common.env_checker import check_env


class RCLTspiceEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # 1. 電路規格
        self.target_v = 3.0
        self.target_time = 0.020
        self.max_steps = 10

        self.actions = {0: -100, 1: 0, 2: 100}

        self.action_space = spaces.Discrete(3)

        # 與之前 PPO 模型相同的 State 格式
        self.observation_space = spaces.Box(
            low=np.array([0.1, 0, 3], dtype=np.float32),
            high=np.array([1.0, 5, 3], dtype=np.float32),
            dtype=np.float32
        )

        self.resistor = 2000
        self.steps = 0

        # 2. 設定 LTspice
        folder = Path(__file__).resolve().parent
        self.circuit = folder / "RC_test.asc"

        ltspice_path = r"D:\LTspice.exe"
        my_ltspice = LTspice.create_from(ltspice_path)

        self.runner = SimRunner(
            simulator=my_ltspice,
            output_folder=str(folder / "simulation_results")
        )

        # 儲存已經模擬過的電阻結果
        self.cache = {}

    # 3. 使用 LTspice 計算 Vout
    def calculate_vout(self):

        if self.resistor in self.cache:
            return self.cache[self.resistor]

        netlist = SpiceEditor(str(self.circuit))

        netlist.set_component_value(
            "R1", str(self.resistor)
        )

        raw_file, log_file = self.runner.run_now(
            netlist,
            run_filename=f"RL_R{self.resistor}.net"
        )

        if raw_file is None:
            raise RuntimeError("LTspice 模擬失敗")

        raw = RawRead(str(raw_file))

        time = raw.get_trace("time").get_wave()
        vout = raw.get_trace("V(vout)").get_wave()

        vout_20ms = float(
            np.interp(self.target_time, time, vout)
        )

        self.cache[self.resistor] = vout_20ms
        return vout_20ms

    # 4. 取得 State
    def get_state(self):
        return np.array(
            [
                self.resistor / 5000,
                self.calculate_vout(),
                self.target_v
            ],
            dtype=np.float32
        )

    # 5. 重設環境
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) #Seed 用來讓隨機結果可以實現

        if options is not None and "initial_resistor" in options:
            self.resistor = int(options["initial_resistor"])
        if not 500 <= self.resistor <= 5000:
            raise ValueError("初始電阻必須在 500~5000Ω")
        else:
            self.resistor = int(
            self.np_random.integers(15, 28) * 100 #隨機產生初始電阻(15~27)*100
        )
        self.steps = 0
        return self.get_state(), {}

    # 6. 執行 Action
    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError("無效的 Action")

        self.resistor += self.actions[int(action)]
        self.resistor = int(
            np.clip(self.resistor, 500, 5000)
        )

        self.steps += 1

        vout = self.calculate_vout()
        reward = -abs(vout - self.target_v)

        terminated = bool(reward >= -0.1)
        truncated = bool(self.steps >= self.max_steps)

        state = self.get_state()

        info = {
            "resistor": self.resistor,
            "vout": vout
        }

        return state, float(reward), terminated, truncated, info


# 7. 測試新環境
if __name__ == "__main__":

    env = RCLTspiceEnv()
    check_env(env)
    print("LTspice Gymnasium 環境檢查成功！")

    state, info = env.reset(seed=42)
    print("初始電阻:", env.resistor)
    print("初始 State:", state)

    state, reward, terminated, truncated, info = env.step(2)

    print("修改後 R1:", env.resistor)
    print("Vout:", info["vout"])
    print("Reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)

    env.close()
