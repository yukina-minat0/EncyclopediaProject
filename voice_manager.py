import asyncio
import edge_tts
import pygame
import threading
import os
import time


class VoiceManager:
    def __init__(self):
        # 初始化音频播放器
        pygame.mixer.init()

        # 定义可选声音列表
        self.voices = {
            "male": "zh-CN-YunxiNeural",  # 阳光男声
            "female": "zh-CN-XiaoxiaoNeural",  # 温柔女声
            "yukkuri": "yukkuri_mode"  # 预留给油库里
        }
        self.current_mode = "female"  # 默认女声
        self.is_playing = False

    def _play_worker(self, file_path):
        """内部函数：在独立线程中执行播放"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"播放出错: {e}")
        finally:
            self.is_playing = False

    async def _amain(self, text, voice):
        """内部异步函数：调用 Edge-TTS 生成语音文件"""
        output_file = "temp_voice.mp3"
        # 这里的 text 最好限制长度，百科太长建议截断
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

        # 生成后，开启线程播放，不阻塞异步循环
        threading.Thread(target=self._play_worker, args=(output_file,), daemon=True).start()

    def speak(self, text):
        """对外接口：根据当前模式说话"""
        if self.current_mode == "yukkuri":
            # 💡 这里是你之后接入 AquesTalk DLL 的地方
            print(f"【油库里彩蛋触发】: {text}")
            return

        if self.is_playing:
            pygame.mixer.music.stop()  # 如果正在说，就掐断说新的

        self.is_playing = True
        voice_type = self.voices[self.current_mode]

        # 开启新线程运行异步任务，防止主界面卡死
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._amain(text, voice_type))
            loop.close()

        threading.Thread(target=run_async, daemon=True).start()

    def set_mode(self, mode_name):
        if mode_name in self.voices:
            self.current_mode = mode_name
            print(f">>> 语音模式切换为: {mode_name}")