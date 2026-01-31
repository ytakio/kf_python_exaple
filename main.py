import numpy as np
import matplotlib.pyplot as plt
from ekf_filter import QuaternionEKF

dt = 0.01
steps = 1000
times = np.arange(steps) * dt

# --- シミュレーションデータ生成 (Roll回転) ---
true_roll_deg = np.zeros(steps)
# 2秒後から30度まで回転して停止
true_roll_deg[200:500] = np.linspace(0, 30, 300)
true_roll_deg[500:] = 30

# ジャイロ (X軸): 真値の微分 + バイアス + ノイズ
gyro_bias = 0.5
gyro_raw = np.gradient(true_roll_deg, dt) + gyro_bias + np.random.normal(0, 0.1, steps)

# 加速度 (Y, Z軸): Roll回転に応じた重力ベクトルの変化
# Roll=0 -> Acc=[0, 1] (Z軸のみ)
# Roll=90 -> Acc=[1, 0] (Y軸のみ)
true_roll_rad = np.radians(true_roll_deg)
acc_y_clean = np.sin(true_roll_rad)
acc_z_clean = np.cos(true_roll_rad)

# ノイズを付加
acc_noise = 0.05
acc_y_raw = acc_y_clean + np.random.normal(0, acc_noise, steps)
acc_z_raw = acc_z_clean + np.random.normal(0, acc_noise, steps)

# --- フィルタ実行 ---
ekf = QuaternionEKF(dt)
est_roll = []
est_bias = []

for i in range(steps):
    ekf.predict(np.radians(gyro_raw[i]))
    ekf.update(acc_y_raw[i], acc_z_raw[i])

    est_roll.append(ekf.get_euler_roll())
    est_bias.append(np.degrees(ekf.x[4]))

# --- 可視化 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 姿勢推定結果
ax1.plot(times, true_roll_deg, "k--", linewidth=2, label="True Roll")
ax1.plot(
    times, np.cumsum(gyro_raw) * dt, "b", alpha=0.3, label="Gyro Integration (Drifting)"
)
ax1.plot(times, est_roll, "r", linewidth=2, label="EKF Estimated")
ax1.set_title("Sensor Fusion: X-Axis Roll Estimation")
ax1.set_ylabel("Roll Angle [deg]")
ax1.legend()
ax1.grid(True)

# バイアス推定結果
ax2.plot(times, [gyro_bias] * steps, "k--", label="True Bias")
ax2.plot(times, est_bias, "b", linewidth=2, label="Estimated Bias")
ax2.set_title(f"Gyro Bias Estimation (True={gyro_bias} deg/s)")
ax2.set_ylabel("Bias [deg/s]")
ax2.set_xlabel("Time [s]")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
