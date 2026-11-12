import numpy as np
class RC_Env:
    #1. 初始化環境
    def __init__(self):
        self.capacitance = 10e-6
        self.target_v = 3.0
        self.target_t = 0.02
        self.max_steps = 10
        self.action = {0: -100, 1: 0, 2: 100}
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
    def reset(self):
        self.resistor = 2000
        self.steps = 0
        return self.get_state()
    #5. 執行Action
    def step(self, action):
        self.resistor += self.action[action]
        self.resistor = int(np.clip(self.resistor, 500, 5000)) #限制電阻值在500~5000之間
        vout_20ms = self.calculate_vout()
        reward = -abs(vout_20ms - self.target_v)
        self.steps += 1
        terminated = reward >= -0.1
        truncated = self.steps >= self.max_steps
        state = self.get_state()
        return state, reward, terminated, truncated
#6. 建立環境物件
env = RC_Env()
#7. 重設環境
state = env.reset()
print("初始State =", state)

#8. 測試一個Action
actions = [1,1,1,1,1]
for action in actions:
    state, reward, terminated, truncated = env.step(action)

    print("-----------------------------")
    print("Action =" ,action) 
    print("State =" ,state)
    print("Reward = ",round(reward, 3))
    print("Terminated = ", terminated)
    if terminated or truncated:
        print("模擬結束！")
        break
#--------------------------------------------------------------------------
#self是指我站存在這環境的數值,假設我在其他環境也有self,
# 那我在其他環境的self就不會影響到這個環境的self,呼叫時只要更改呼叫的環境名稱就好