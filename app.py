import gradio as gr
from main_agent import DeepSeekMultimodalAgent
from signal_engine import run_signal_demo
from cv_engine import run_cv_onnx_demo

agent = DeepSeekMultimodalAgent()

def run_dynamic_agent(input_freq, noise_level):
    # 1. 动态运行信号滤波引擎，算出真实频域主频
    signal_result = run_signal_demo(target_freq=input_freq, noise_level=noise_level)
    
    # 2. 运行端侧视觉推理
    cv_result = run_cv_onnx_demo()
    
    # 3. 组装动态多模态数据送入 Agent 决策中枢
    perception_data = {
        "cv_data": cv_result,
        "signal_data": signal_result
    }
    
    report = agent.reason_and_decide(perception_data)
    
    cv_info = f"ONNX 视觉模型状态: {cv_result['status']}\nTensor 形状: {cv_result['output_shape']}"
    signal_info = f"提取滤波后主频: {signal_result['main_freq_hz']} Hz\n输入噪声设定强度: {noise_level}"
    
    return cv_info, signal_info, report

# 搭建带有动态参数调优的 UI 界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 动态多模态端侧 AI Agent 工业巡检系统")
    gr.Markdown("拖动下方传感器控制参数，体验**真实时滤波计算**与 **DeepSeek 动态多模态推理**。")
    
    with gr.Row():
        with gr.Column():
            freq_slider = gr.Slider(minimum=5.0, maximum=80.0, value=10.0, step=1.0, label="🎛️ 模拟传感器输入主频 (Hz)")
            noise_slider = gr.Slider(minimum=0.0, maximum=3.0, value=0.8, step=0.1, label="🔊 模拟环境高频噪声强度")
            run_btn = gr.Button("⚡ 触发 Agent 实时协同感知与决策", variant="primary")
            
        with gr.Column():
            cv_box = gr.Textbox(label="👁️ 视觉感知模块输出 (CV Engine)")
            signal_box = gr.Textbox(label="👂 信号降噪与 FFT 实时计算 (Signal Engine)")
            
    with gr.Row():
        report_box = gr.Textbox(label="🧠 DeepSeek Agent 动态生成的诊断报告", lines=6)
        
    run_btn.click(
        fn=run_dynamic_agent,
        inputs=[freq_slider, noise_slider],
        outputs=[cv_box, signal_box, report_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)