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
        try:
            # 默认以 8000Hz 初始化，匹配 Yukkuri 采样率
            pygame.mixer.init(frequency=8000)
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

        # 字母及数字发音拟音表
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
            print(f"✅ 语音引擎就绪 (WanaKana 自动适配模式)")
        except:
            self.aq_lib = None
            print("⚠️ AquesTalk DLL 加载失败")

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
        """核心转换逻辑"""
        py_results = pinyin(text, style=Style.NORMAL, errors='default')
        kana_out_list = []

        # 🟢 自动获取转换函数（适配不同版本的 wanakana 库）
        to_kana_fn = getattr(wanakana, 'to_katakana', getattr(wanakana, 'to_kana', None))

        for item in py_results:
            token = item[0].lower()

            # 1. 查本地字典
            if token in self.kana_dict:
                kana_out_list.append(self.kana_dict[token])
            # 2. 处理英文和数字
            elif re.match(r'[a-z0-9.]+', token):
                if len(token) == 1 or re.match(r'^[0-9.]+$', token):
                    for char in token:
                        kana_out_list.append(self.alpha_map.get(char, ""))
                elif to_kana_fn:
                    # 如果是单词，尝试让 wanakana 转换
                    kana_out_list.append(to_kana_fn(token))
                else:
                    kana_out_list.append(token)
            else:
                kana_out_list.append(token)

        combined_text = "".join(kana_out_list)
        # 再次确认通过假名转换
        final_kana = to_kana_fn(combined_text) if to_kana_fn else combined_text

        # 净化正则
        pattern = r'[\u3040-\u309F\u30A0-\u30FF\u30FC\u3001\u3002\uFF01\uFF1F]'
        return "".join(re.findall(pattern, final_kana))

    def speak(self, text):
        """语音播放入口"""
        self.stop_event.clear()
        if not text or self.is_playing: return
        self.is_playing = True

        # 🟢 过滤百科开头废话，从第一个标题处开始读
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
            sentences = [s.strip() for s in re.split(r'[，。！？\n,;!?:()]', text) if s.strip()]
            for s in sentences:
                if self.stop_event.is_set(): break
                kana_str = self._pinyin_to_kana(s) + "。"
                if not kana_str.strip("。"): continue

                encoded_data = kana_str.encode('shift-jis', errors='ignore') + b'\x00'
                size = ctypes.c_int(0)
                wav_ptr = self.aq_lib.AquesTalk_Synthe(encoded_data, 100, ctypes.byref(size))

                if wav_ptr:
                    wav_data = ctypes.string_at(wav_ptr, size.value)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_file = tf.name
                        tf.write(wav_data)
                    self.active_temp_files.add(temp_file)
                    self.aq_lib.AquesTalk_FreeWave(wav_ptr)
                    self._play_worker(temp_file, freq=8000)
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
                if not self.stop_event.is_set(): self._play_worker(output, freq=44100)
            finally:
                self._cleanup_file(output)
                self.is_playing = False

        asyncio.run(_task())

    def _play_worker(self, file_path, freq=44100):
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=freq)
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
        except:
            pass

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
                pygame.mixer.music.unload()
                pygame.mixer.quit()
        except:
            pass
        for path in list(self.active_temp_files): self._cleanup_file(path)

    def set_mode(self, mode_name):
        if mode_name in self.voices: self.current_mode = mode_name