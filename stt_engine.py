import speech_recognition as sr
import whisper
import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal


class STTEngine(QThread):
    """
    独立语音转文字引擎模块
    状态流转：就绪 -> 正在聆听 -> 正在识别 -> 完成
    """
    # 定义状态信号，方便 UI 监听
    status_changed = pyqtSignal(str)  # 传递状态文字
    result_ready = pyqtSignal(str)  # 传递识别出的文本

    def __init__(self, model_size="base"):
        super().__init__()
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        try:
            # 1. 延迟加载模型（仅在第一次启动线程时加载）
            if self.model is None:
                self.status_changed.emit("正在初始化 AI 语音模型...")
                # 使用 FP16 可以在有显卡时提速，CPU 建议 FP32
                self.model = whisper.load_model(self.model_size, device=self.device)

            # 2. 配置录音设备
            r = sr.Recognizer()
            # 这里的参数可以根据你的环境微调
            r.dynamic_energy_threshold = True

            with sr.Microphone() as source:
                # 环境噪音校准
                self.status_changed.emit("正在消除背景杂音...")
                r.adjust_for_ambient_noise(source, duration=0.8)

                self.status_changed.emit("🎤 正在聆听，请提问...")
                # phrase_time_limit 限制单次录音最长时间
                audio = r.listen(source, phrase_time_limit=15)

            # 3. 转换为 Whisper 格式并识别
            self.status_changed.emit("🧠 正在精准翻译...")

            # 将音频字节流转为浮点数数组
            audio_data = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0

            # 执行识别
            result = self.model.transcribe(audio_data, language="zh")
            text = result.get("text", "").strip()

            if text:
                self.result_ready.emit(text)
                self.status_changed.emit("✅ 识别成功")
            else:
                self.status_changed.emit("❓ 未能识别出文字")

        except Exception as e:
            self.status_changed.emit(f"❌ 语音转换失败: {str(e)}")