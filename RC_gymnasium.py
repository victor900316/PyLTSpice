import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import numpy as np
class RC_Env(gym.Env):
    #1. 初始化環境
    def __init__(self):
        self.capacitance = 10e-6
        self.target_v = 3.0
        self.target_t = 0.02
        self.max_steps = 10
        self.action = {0: -100, 1: 0, 2: 100}
    #定義 Action Space
        self.action_space = spaces.Discrete(3)

    #定義 Observation Space
        self.observation_space = spaces.Box(
            low=np.array([500, 0, 3], dtype=np.float32),
            high=np.array([5000, 5, 3], dtype=np.float32),
            dtype=np.float32
        )    
        self.resistor = 2000
        self.steps = 0

    #2. 計算RC電路的輸出電壓
    def calculate_vout(self):
        vout_20ms = 5 * (1 - np.exp(-self.target_t / (self.resistor * self.capacitance))) #RC充電公式
        return vout_20ms

    #3. 取得目前 State
    def get_state(self):
        vout_20ms = self.calculate_vout()
        return np.array([self.resistor, vout_20ms, self.target_v],
        dtype=np.float32                        
        )
    #4. 重設環境
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resistor = 2000
        self.steps = 0
        state = self.get_state()
        info = {}
        return state, info
    #5. 執行Action
    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError("無效的 Action")
        
        self.resistor += self.action[action]
        self.resistor = int(
            np.clip(self.resistor, 500, 5000)  #限制電阻值在500~5000之間
        ) 
        vout_20ms = self.calculate_vout()
        reward = -abs(vout_20ms - self.target_v)
        self.steps += 1
        terminated = bool(reward >= -0.1)
        truncated = bool(self.steps >= self.max_steps)
        state = self.get_state()

        info = {
            "resistor": self.resistor,
            "vout": vout_20ms
        }
        return state, float(reward), terminated, truncated, info
#6. 建立環境物件
env = RC_Env()
#檢查環境
check_env(env, skip_render_check=True)
print("Gymnasium 環境檢查成功！")
#7. 重設環境
state, info = env.reset()
print("初始 State =", state)

#8. 測試一個Action
actions = [0,2,2]
for action in actions:
    state, reward, terminated, truncated, info = env.step(action)

    print("-----------------------------")
    print("Action =" ,action) 
    print("State =" ,state)
    print("Reward = ",round(reward, 3))
    print("Terminated = ", terminated)
    print("truncated", truncated)
    print("info", info)
    if terminated or truncated:
        print("模擬結束！")
        break
env.close()
#--------------------------------------------------------------------------
#self是指我站存在這環境的數值,假設我在其他環境也有self,
# 那我在其他環境的self就不會影響到這個環境的self,呼叫時只要更改呼叫的環境名稱就好