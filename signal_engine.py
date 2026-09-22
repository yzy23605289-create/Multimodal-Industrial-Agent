import numpy as np
from scipy import signal

def run_signal_demo(target_freq=10.0, noise_level=0.8):
    print(f"\n[1/3] 正在生成传感器信号 (设定主频: {target_freq}Hz, 噪声强度: {noise_level})...")
    fs = 1000  # 采样频率 1000Hz
    t = np.linspace(0, 1.0, fs, endpoint=False)
    
    # 动态构建信号：指定的目标主频 + 50Hz 工频干扰 + 动态高斯噪声
    clean_sig = np.sin(2 * np.pi * target_freq * t)
    noise = noise_level * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.2, fs)
    raw_signal = clean_sig + noise

    print("[2/3] 使用 SciPy 巴特沃斯低通滤波器进行降噪滤波...")
    b, a = signal.butter(4, 35, fs=fs, btype='low')  # 截止频率设为 35Hz
    filtered_sig = signal.filtfilt(b, a, raw_signal)

    print("[3/3] 执行快速傅里叶变换 (FFT) 频域解析...")
    fft_vals = np.abs(np.fft.fft(filtered_sig))[:fs//2]
    freqs = np.fft.fftfreq(fs, 1/fs)[:fs//2]
    main_freq = freqs[np.argmax(fft_vals)]

    print(f"🎉 FFT 精准提取出真实主频: {main_freq:.1f} Hz\n")
    return {"status": "Success", "main_freq_hz": float(main_freq)}

if __name__ == "__main__":
    run_signal_demo()