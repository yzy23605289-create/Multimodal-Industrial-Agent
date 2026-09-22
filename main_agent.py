import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 导入本地视觉与信号感知引擎
from cv_engine import run_cv_onnx_demo
from signal_engine import run_signal_demo

# 加载 .env 文件中的 API Key
load_dotenv()

class DeepSeekMultimodalAgent:
    def __init__(self, agent_name="Challenger-DeepSeek-Agent"):
        self.agent_name = agent_name
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        
        # 初始化 OpenAI/DeepSeek 客户端
        if self.api_key and not self.api_key.startswith("sk-your"):
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            self.use_llm = True
            print(f"=== 🚀 初始化 {self.agent_name} (DeepSeek 驱动) ===")
        else:
            self.use_llm = False
            print(f"=== 🚀 初始化 {self.agent_name} (离线规则回退模式 - 请配置 .env 中的 DEEPSEEK_API_KEY) ===")

    def perceive_multimodal_data(self):
        print("\n--- 📡 阶段一：启动多模态端侧感知引擎 ---")
        cv_result = run_cv_onnx_demo()
        signal_result = run_signal_demo()
        return {
            "cv_data": cv_result,
            "signal_data": signal_result
        }

    def reason_and_decide(self, perception_data):
        print("\n--- 🧠 阶段二:Agent 决策中枢进行多模态数据融合与 LLM 推理 ---")
        
        cv_shape = perception_data['cv_data']['output_shape']
        freq = perception_data['signal_data']['main_freq_hz']
        
        # 构造给 DeepSeek 的提示词 (Prompt)
        prompt = f"""你是一个智能工业巡检与设备故障诊断专家 AI Agent。
请根据传感器与端侧模型传入的多模态特征数据，生成一份结构严密、专业的设备运行诊断报告。

【输入多模态数据】
1. 视觉侧(ONNX Runtime 引擎检测): 节点输出 Tensor 形状为 {cv_shape}，捕获到设备外观关键特征。
2. 信号侧(SciPy 巴特沃斯低通滤波 + FFT 频域变换): 成功过滤掉 50Hz 高频电磁噪声，提取到核心主频为 {freq} Hz。

【输出要求】
1. 评估设备当前运行状态（正常/预警/异常）。
2. 结合视觉特征与 {freq} Hz 频域特征给出科学解释。
3. 给出后续运维建议。
字数控制在 200 字左右，语言严谨专业。
"""

        if self.use_llm:
            try:
                print("⏳ 正在请求 DeepSeek API 进行深度逻辑思考...")
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的工业多模态 AI Agent 诊断专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ API 调用异常: {e}，自动切换至本地专家逻辑")
                return self._fallback_reasoning(cv_shape, freq)
        else:
            return self._fallback_reasoning(cv_shape, freq)

    def _fallback_reasoning(self, cv_shape, freq):
        time.sleep(1)
        return (
            f"【DeepSeek 仿真诊断报告】\n"
            f"1. 视觉评估:ONNX 提取输出维度 {cv_shape}，设备部件位置无物理位移或明显破损。\n"
            f"2. 信号评估:经过巴特沃斯滤波后,FFT 主频稳定在 {freq} Hz,处于正常工频区间(10Hz 额定转速)。\n"
            f"3. 综合结论：设备健康度 98%。高频噪声已成功被端侧滤波算法屏蔽，当前无隐患，建议保持正常巡检周期。"
        )

if __name__ == "__main__":
    agent = DeepSeekMultimodalAgent()
    raw_data = agent.perceive_multimodal_data()
    final_report = agent.reason_and_decide(raw_data)
    
    print("\n" + "="*60)
    print("📋 【DeepSeek Agent 最终多模态诊断分析】")
    print("="*60)
    print(final_report)
    print("="*60 + "\n")