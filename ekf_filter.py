import numpy as np


class QuaternionEKF:
    def __init__(self, dt, q_init=[1.0, 0.0, 0.0, 0.0], bias_init=0.0):
        self.dt = dt
        self.x = np.array([*q_init, bias_init], dtype=float)

        # 【秘策1】初期の不確実性を操作
        # 姿勢(前半4つ)はそこそこ自信ある(1e-2)が、
        # バイアス(最後)は全く自信がない(1.0)と設定。
        # -> これにより、開始0.1秒でバイアスを一気に修正しに行きます。
        self.P = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1.0])

        # 【秘策2】プロセスノイズの引き締め
        # バイアス(x[4])の変動要因を 1e-10 まで下げます。
        # 「一度決まったバイアスは、めったなことでは動かない」と信じ込ませます。
        self.Q = np.diag([1e-7, 1e-7, 1e-7, 1e-7, 1e-10])

        # 【秘策3】観測ノイズの緩和
        # 加速度のノイズをもう少し「あるものだ」と許容させます。
        # これでジタバタしなくなります。
        self.R = np.eye(2) * 0.5

    def predict(self, gyro_x):
        qw, qx, qy, qz, b = self.x
        omega = gyro_x - b

        # X軸回転(Roll)のクォータニオン微分
        # dq/dt = 0.5 * q * [0, w, 0, 0]
        half_omega_dt = 0.5 * omega * self.dt
        new_qw = qw - qx * half_omega_dt
        new_qx = qx + qw * half_omega_dt
        # Y, Z成分はX軸回転のみなら理論上0に近いが、計算上維持
        new_qy = qy + qz * half_omega_dt
        new_qz = qz - qy * half_omega_dt

        self.x[0] = new_qw
        self.x[1] = new_qx
        self.x[2] = new_qy
        self.x[3] = new_qz
        self.x[:4] /= np.linalg.norm(self.x[:4])

        # ヤコビ行列 F (Roll回転用)
        F = np.eye(5)
        # d(dq)/d(bias) の項
        F[0, 4] = 0.5 * qx * self.dt
        F[1, 4] = -0.5 * qw * self.dt
        F[2, 4] = -0.5 * qz * self.dt
        F[3, 4] = 0.5 * qy * self.dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, acc_y, acc_z):
        qw, qx, qy, qz, _ = self.x

        # 重力ベクトル[0,0,1]を機体座標系へ変換 (Y, Z成分を観測)
        # gy = 2(yz + wx)
        # gz = 1 - 2(xx + yy)
        hy = 2.0 * (qy * qz + qw * qx)
        hz = 1.0 - 2.0 * (qx * qx + qy * qy)
        z_pred = np.array([hy, hz])

        # 観測ヤコビ行列 H (2x5)
        # H[0] -> d(hy)/dq
        # H[1] -> d(hz)/dq
        H = np.zeros((2, 5))
        H[0, 0] = 2 * qx
        H[0, 1] = 2 * qw
        H[0, 2] = 2 * qz
        H[0, 3] = 2 * qy
        H[1, 0] = 0
        H[1, 1] = -4 * qx
        H[1, 2] = -4 * qy
        H[1, 3] = 0

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 残差 (観測値 - 予測値)
        # 加速度ベクトルは正規化して使うのが一般的
        acc = np.array([acc_y, acc_z])
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 1e-6:
            z_meas = acc / acc_norm
            y = z_meas - z_pred

            self.x = self.x + K @ y
            self.x[:4] /= np.linalg.norm(self.x[:4])
            self.P = (np.eye(5) - K @ H) @ self.P

    def get_euler_roll(self):
        qw, qx, qy, qz, _ = self.x
        # Roll角 (X軸回転)
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        return np.degrees(roll)
