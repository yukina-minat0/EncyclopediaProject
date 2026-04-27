import ctypes
import os
import pygame
import threading
import asyncio
import edge_tts
import re
import tempfile
import time
import wanakana
from pypinyin import pinyin, Style


class VoiceManager:
    def __init__(self):
        # 统一使用标准的 44100Hz，避免频繁 init/quit 导致爆音
        try:
            pygame.mixer.init(frequency=44100)
        except Exception as e:
            print(f"音频初始化提示: {e}")

        self.voices = {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural", "yukkuri": "yukkuri"}
        self.current_mode = "female"
        self.is_playing = False
        self.stop_event = threading.Event()
        self.active_temp_files = set()

        # 加载本地映射字典
        self.dict_file = "mapping.tsv"
        self.kana_dict = self._load_mapping_tsv(self.dict_file)

        # 字母及数字发音拟音表 (Yukkuri 专用)
        self.alpha_map = {
            'a': 'ee', 'b': 'bii', 'c': 'shii', 'd': 'dei', 'e': 'ii',
            'f': 'efu', 'g': 'jii', 'h': 'ecchi', 'i': 'ai', 'j': 'jee',
            'k': 'kee', 'l': 'eru', 'm': 'emu', 'n': 'enu', 'o': 'oo',
            'p': 'pii', 'q': 'kyuu', 'r': 'aaru', 's': 'esu', 't': 'tei',
            'u': 'yuu', 'v': 'bui', 'w': 'daburyuu', 'x': 'ekkusu', 'y': 'wai', 'z': 'zetto',
            '0': 'zero', '1': 'ichi', '2': 'ni', '3': 'san', '4': 'yon', '5': 'go', '6': 'roku', '7': 'nana',
            '8': 'hachi', '9': 'kyuu', '.': 'ten'
        }

        self.dll_path = "./AquesTalk.dll"
        try:
            self.aq_lib = ctypes.cdll.LoadLibrary(self.dll_path)
            self.aq_lib.AquesTalk_Synthe.restype = ctypes.c_void_p
            self.aq_lib.AquesTalk_Synthe.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            self.aq_lib.AquesTalk_FreeWave.argtypes = [ctypes.c_void_p]
            print(f"✅ 语音引擎就绪 (原生 DLL 模式)")
        except:
            self.aq_lib = None
            print("⚠️ AquesTalk DLL 加载失败，油库里模式将不可用")

    def _load_mapping_tsv(self, filepath):
        mapping = {}
        if not os.path.exists(filepath): return mapping
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2: mapping[parts[0].lower()] = parts[1]
        except:
            pass
        return mapping

    def _pinyin_to_kana(self, text):
        """将中文/英文转换为 Yukkuri 识别的假名符号流"""
        py_results = pinyin(text, style=Style.NORMAL, errors='default')
        kana_out_list = []
        to_kana_fn = getattr(wanakana, 'to_katakana', getattr(wanakana, 'to_kana', None))

        for item in py_results:
            token = item[0].lower()
            if token in self.kana_dict:
                kana_out_list.append(self.kana_dict[token])
            elif re.match(r'[a-z0-9.]+', token):
                if len(token) == 1 or re.match(r'^[0-9.]+$', token):
                    for char in token: kana_out_list.append(self.alpha_map.get(char, ""))
                elif to_kana_fn:
                    kana_out_list.append(to_kana_fn(token))
            else:
                # 尝试通过 wanakana 兜底处理无法识别的部分
                if to_kana_fn: kana_out_list.append(to_kana_fn(token))

        combined_text = "".join(kana_out_list)
        # 净化逻辑：只保留假名和基础标点，避免 DLL 崩溃
        pattern = r'[\u3040-\u309F\u30A0-\u30FF\u30FC\u3001\u3002\uFF01\uFF1F]'
        return "".join(re.findall(pattern, combined_text))

    def speak(self, text):
        """语音播放入口，支持打断"""
        if not text: return

        # 打断当前正在播放的声音
        self.stop_event.set()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

        # 给一点时间让旧线程退出
        time.sleep(0.1)
        self.stop_event.clear()
        self.is_playing = True

        # 过滤 Markdown 和多余换行
        start_idx = text.find("###")
        final_content = text[start_idx:] if start_idx != -1 else text
        clean_msg = re.sub(r'[#\*]', '', final_content)
        clean_msg = re.sub(r'\n+', '。', clean_msg)

        if self.current_mode == "yukkuri" and self.aq_lib:
            threading.Thread(target=self._yukkuri_process, args=(clean_msg,), daemon=True).start()
        else:
            voice = self.voices.get(self.current_mode, self.voices["female"])
            threading.Thread(target=self._run_edge_tts, args=(clean_msg, voice), daemon=True).start()

    def _yukkuri_process(self, text):
        try:
            # 句子切分，避免单次合成过长
            sentences = [s.strip() for s in re.split(r'[，。！？\n,;!?:()]', text) if s.strip()]
            for s in sentences:
                if self.stop_event.is_set(): break

                kana_str = self._pinyin_to_kana(s)
                if not kana_str: continue

                # AquesTalk 合成参数：100 为基准语速，建议 120-130 更有油库里味
                encoded_data = (kana_str + "。").encode('shift-jis', errors='ignore') + b'\x00'
                size = ctypes.c_int(0)
                wav_ptr = self.aq_lib.AquesTalk_Synthe(encoded_data, 100, ctypes.byref(size))

                if wav_ptr:
                    wav_data = ctypes.string_at(wav_ptr, size.value)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_file = tf.name
                        tf.write(wav_data)

                    self.active_temp_files.add(temp_file)
                    self.aq_lib.AquesTalk_FreeWave(wav_ptr)

                    # 播放
                    self._play_worker(temp_file)
                    self._cleanup_file(temp_file)
        finally:
            self.is_playing = False

    def _run_edge_tts(self, text, voice):
        async def _task():
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                output = tf.name
            self.active_temp_files.add(output)
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output)
                if not self.stop_event.is_set():
                    self._play_worker(output)
            finally:
                self._cleanup_file(output)
                self.is_playing = False

        # 解决 asyncio.run 可能在线程中冲突的问题
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(_task())

    def _play_worker(self, file_path):
        """通用的播放执行器，无需重启 mixer"""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"播放异常: {e}")

    def _cleanup_file(self, path):
        try:
            if os.path.exists(path): os.remove(path)
            if path in self.active_temp_files: self.active_temp_files.remove(path)
        except:
            pass

    def quit(self):
        self.stop_event.set()
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except:
            pass
        for path in list(self.active_temp_files): self._cleanup_file(path)

    def set_mode(self, mode_name):
        if mode_name in self.voices:
            self.current_mode = mode_name
            print(f"🎤 语音模式切换至: {mode_name}")