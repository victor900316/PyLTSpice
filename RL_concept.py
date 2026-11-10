import numpy as np
resistor = 2000
target_v = 3.0
capacitance = 10e-6
#2. 定義Action執行的動作
actions = {
    0: -500,
    1: 0,
    2: 500 
}
#3. 定義Action
action = 0
#4. 根據Action 修改電阻
resistor = resistor + actions[action]
resistor = np.clip(resistor, 500, 5000) #限制電阻值在500~5000之間 np.clip(數值,最小值,最大值)
#5. 計算20ms的理論輸出電壓
time = 0.02
vout_20ms = 5 * (1 - np.exp(-time / (resistor * capacitance))) #RC充電公式
#6. 計算Reward值
reward = -abs(vout_20ms - target_v)  #計算reward值，越接近目標值越好
#7. 建立新的State
state = [resistor, vout_20ms,target_v]

print("Action =", action)
print("新的Resistor =",resistor,"ohm")
print("Vout =",round(vout_20ms, 3),"v")
print ("Reward =",round(reward, 3))
print("新的State =",state)
