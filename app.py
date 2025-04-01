import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from numba import jit
import time  # timeモジュールをインポート

# JITコンパイルされたラプラシアン計算関数（簡素化）
@jit(nopython=True)
def compute_laplacian(Pz, dx, dy, dz):
    laplacian_Pz = np.zeros_like(Pz)
    for i in range(1, Pz.shape[0] - 1, 2):  # 2ステップ飛ばし
        for j in range(1, Pz.shape[1] - 1, 2):  # 2ステップ飛ばし
            for k in range(1, Pz.shape[2] - 1, 2):  # 2ステップ飛ばし
                laplacian_Pz[i, j, k] = (
                    (Pz[i + 1, j, k] - 2 * Pz[i, j, k] + Pz[i - 1, j, k]) / dx**2 +
                    (Pz[i, j + 1, k] - 2 * Pz[i, j, k] + Pz[i, j - 1, k]) / dy**2 +
                    (Pz[i, j, k + 1] - 2 * Pz[i, j, k] + Pz[i, j, k - 1]) / dz**2
                )
    return laplacian_Pz

# タイトル
st.title("Phase-field Simulation")

# 説明
st.write("A simple phase field method is used to calculate the time evolution of ferroelectric vortex structures. If the calculation time is too long, reduce the grid size.")

# パラメータ入力フォーム
Nx = st.number_input("Nx (gridsize X)", min_value=10, max_value=250, value=100, step=10)
Ny = st.number_input("Ny (gridsize Y)", min_value=10, max_value=250, value=100, step=10)
Nz = st.number_input("Nz (gridsize Z)", min_value=10, max_value=250, value=125, step=10)

P0 = st.number_input("P0 (Initial Amplitude of Polarization)", min_value=0.0, max_value=1.0, value=0.5)
lambda_z = st.number_input("λz (Attenuation parameter in z direction)", min_value=0.0, max_value=10.0, value=1.0)

dt = st.number_input("dt (Time step size)", min_value=0.0001, max_value=1.0, value=0.05, format="%.4f")
steps = st.number_input("Number of calculation steps", min_value=10, max_value=10000, value=500, step=100)

temperature = st.number_input("temperature", min_value=0.0, max_value=1000.0, value=300.0)

# 実行ボタン
if st.button("Start simulation"):
    st.write("Simulation running... 🚀")

    # --- オウムのGIFを表示 ---
    parrot_gif = "parrot.gif"  # GIFファイルのパス（アプリのディレクトリ内に配置）
    st.image(parrot_gif, caption="waiting for simulation 🦜🎶", width=300)

    # --- シミュレーションパラメータの設定 ---
    dx = dy = dz = 0.4  # nm
    Lx, Ly = Nx * dx, Ny * dy
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Nz * dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # --- 初期条件の設定（ボルテックス構造） ---
    Pz = P0 * np.sin(2 * np.pi * X / Lx + 2 * np.pi * Y / Ly) * np.exp(-Z / lambda_z)

    # 計算開始時間を記録
    start_time = time.time()

    # --- 時間発展のループ ---
    for step in range(steps):
        # ラプラシアン計算を高速化
        laplacian_Pz = compute_laplacian(Pz, dx, dy, dz)

        # Allen-Cahn 型の時間発展
        dF_dP = -Pz + Pz**3  # 自由エネルギーの勾配
        Pz += dt * (-dF_dP + laplacian_Pz)  # 時間発展

        # 進行状況の表示と予想残り時間の計算
        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (step + 1)  # 1ステップあたりの計算時間
            remaining_steps = steps - step  # 残りのステップ数
            estimated_time_left = time_per_step * remaining_steps  # 残り時間の予測

            # 残り時間の表示
            st.write(f"Step {step}/{steps} Calculating... (Remaining time prediction: {estimated_time_left / 60:.2f} minutes)")

    # 計算終了時間を記録
    end_time = time.time()

    # 実行時間を計算
    elapsed_time = end_time - start_time
    st.write(f"Simulation completed! ✅ Calculation time: {elapsed_time:.2f} seconds")

    # --- 結果の可視化 (XZ 平面) ---
    y_slice = Ny // 2  # 中央のXZ断面
    Pz_xz = Pz[:, y_slice, :]

    # 結果をプロット
    plt.figure(figsize=(8, 6))
    plt.imshow(Pz_xz.T, origin='lower', cmap='coolwarm', aspect='auto',
               extent=[0, Nx * dx, 0, Nz * dz], vmin=-1, vmax=1)

    plt.xlabel("X (nm)")
    plt.ylabel("Z (nm)")
    plt.colorbar(label="Polarization (arb. units)")
    plt.title("Phase-Field Simulation (XZ slice)")

    # プロットをStreamlitに表示
    st.pyplot(plt)
