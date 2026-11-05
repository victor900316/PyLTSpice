
from pathlib import Path #負責管理檔案路徑
from PyLTSpice import SimRunner, SpiceEditor, LTspice,  RawRead 
# RawRead:負責模擬與讀取結果
# SimRunner：負責啟動模擬。
# SpiceEditor：負責讀取電路，之後也能修改元件參數。
# LTspice：指定要使用 LTspice 模擬器。
import matplotlib.pyplot as plt #負責繪圖
import numpy as np #負責數值運算

# 1. 找到 Python 和電路檔案
folder = Path(__file__).resolve().parent
circuit = folder / "RC_test.asc"

# 2. 指定 LTspice 執行檔位置（請替換成你的真實路徑）
ltspice_path = r"D:\LTspice.exe"
# 3. 建立模擬器設定
my_ltspice = LTspice.create_from(ltspice_path)

# 4. 建立模擬器
runner = SimRunner(
    simulator=my_ltspice,
    output_folder=str(folder / "simulation_results")
)

# 5. 讀取電路
netlist = SpiceEditor(str(circuit))

#修改原先的R1電阻值(1k改為2k)
#netlist.set_component_value("R1", "2k")

#定義三組要修改的數值
resistor =  ["1k", "2k", "5k"]
target_v = 3.0
target_t = 0.02 
best_reward = float("-inf")
best_r = None  #還未模擬所以先設置為None
# 6. 執行模擬
for  r  in resistor:
    netlist.set_component_value("R1", r)
    raw_file, log_file = runner.run_now(
        netlist,
        run_filename=f"RC_{r}.net"
    )
    if raw_file is None:
        print(f"RC_{r}模擬失敗！")
        continue
    print(f"RC_{r}模擬完成！")

    # 7. 讀取模擬結果
    raw = RawRead(str(raw_file))

    #讀取所需的資料
    time = raw.get_trace("TIME").get_wave()
    vout = raw.get_trace("V(vout)").get_wave()
    vout_20ms = np.interp(target_t, time, vout)  #插值計算在0.02秒時的電壓值
    reward = -abs(vout_20ms - target_v)  #計算reward值，越接近目標值越好
    print(f"R1={r}")
    print(f"20ms時的 Vout = {vout_20ms:.3f} V")
    print(f"reward = {reward:.3f}")
    print("-----------------------------")
    #找出最佳結果
    if reward>best_reward:
        best_reward=reward
        best_r=r
    #繪製結果
    plt.plot(time*1000,vout,  label=f"R1={r}")

print("最佳電阻:", best_r)
print("最佳Reward:", best_reward)
#9.圖表設置
plt.axvline(20, linestyle="--", label="20ms")
plt.axhline(3, linestyle=":", label="Target = 3V")
plt.title("RC Circuit Reward Comparison")
plt.xlabel("Time (ms)")
plt.ylabel("V(out)")
plt.grid(True)
plt.legend()
plt.show() #顯示三條曲線

