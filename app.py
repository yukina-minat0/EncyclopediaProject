import os
import sys
import cv2
import torch
import numpy as np
import time
import ctypes  # 用于设置 Windows 任务栏 ID
from enum import Enum

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# 引擎导入
from vision import VisionEngine
from deepseek_engine import DeepSeekEngine
from anime_engine import AnimeEngine
from voice_manager import VoiceManager

# 限制线程数
torch.set_num_threads(1)


# =========================
# 📦 资源路径转换函数
# =========================
def get_resource_path(relative_path):
    """ 获取资源绝对路径，兼容 PyInstaller 打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class SelectMode(Enum):
    BOX = 0
    POINT = 1


# =========================
# 💬 对话处理线程
# =========================
class ChatThread(QThread):
    answer_ready = pyqtSignal(str)

    def __init__(self, engine, context, question):
        super().__init__()
        self.engine = engine
        self.context = context
        self.question = question

    def run(self):
        try:
            prompt = f"背景资料：{self.context}\n\n问题：{self.question}"
            res = self.engine.get_wiki(prompt)
            self.answer_ready.emit(res)
        except Exception as e:
            self.answer_ready.emit(f"对话出错: {str(e)}")


# =========================
# 📷 摄像头线程
# =========================
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# =========================
# 🎨 自绘画布
# =========================
class ImageCanvas(QWidget):
    area_selected = pyqtSignal(QRect)
    point_clicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.begin = None
        self.end = None
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.setMouseTracking(True)

    def set_image(self, img):
        self.image = img
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        if self.image is None: return

        h, w = self.image.shape[:2]
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)

        scale_w, scale_h = self.width() / w, self.height() / h
        self.scale = min(scale_w, scale_h)
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        self.offset = QPoint((self.width() - new_w) // 2, (self.height() - new_h) // 2)

        painter.drawImage(QRect(self.offset, QSize(new_w, new_h)), qimg)
        if self.begin and self.end:
            painter.setPen(QPen(QColor(0, 255, 127), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.begin, self.end).normalized())

    def mousePressEvent(self, e):
        self.begin = e.position().toPoint()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, e):
        if self.begin:
            self.end = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.begin:
            self.end = e.position().toPoint()
            rect = QRect(self.begin, self.end).normalized()
            if rect.width() < 10:
                self.point_clicked.emit(self.map_to_image(self.end))
            else:
                self.area_selected.emit(self.map_rect(rect))
        self.begin = None
        self.end = None
        self.update()

    def map_to_image(self, p):
        x = int((p.x() - self.offset.x()) / self.scale)
        y = int((p.y() - self.offset.y()) / self.scale)
        return QPoint(x, y)

    def map_rect(self, r):
        x, y = int((r.x() - self.offset.x()) / self.scale), int((r.y() - self.offset.y()) / self.scale)
        w, h = int(r.width() / self.scale), int(r.height() / self.scale)
        return QRect(x, y, w, h)


# =========================
# 🔥 分析线程核心逻辑
# =========================
def get_segment_result(model, img_raw, data, mode, device="cpu"):
    h, w = img_raw.shape[:2]
    img_display = img_raw.copy()
    if mode == SelectMode.BOX:
        x1, y1 = max(0, data.x()), max(0, data.y())
        x2, y2 = min(w, data.right()), min(h, data.bottom())
        ex = 0.3
        wb, hb = x2 - x1, y2 - y1
        x1e, y1e = max(0, int(x1 - wb * ex)), max(0, int(y1 - hb * ex))
        x2e, y2e = min(w, int(x2 + wb * ex)), min(h, int(y2 + hb * ex))
        roi = img_raw[y1e:y2e, x1e:x2e]
    else:
        x, y = np.clip(data.x(), 0, w - 1), np.clip(data.y(), 0, h - 1)
        results = model.predict(source=img_raw, points=[[x, y]], labels=[1], device=device, retina_masks=True,
                                verbose=False)
        masks = results[0].masks.data
        if masks is None or len(masks) == 0: return img_display, img_raw
        mask_areas = sorted([(m.sum(), m) for m in masks], reverse=True, key=lambda x: x[0])
        best = mask_areas[0][1]
        second = mask_areas[1][1] if len(mask_areas) > 1 else None
        combined = best | second if second is not None else best
        mask_np = combined.cpu().numpy()
        ys, xs = np.where(mask_np > 0)
        x1, x2 = max(0, xs.min() - 25), min(w, xs.max() + 25)
        y1, y2 = max(0, ys.min() - 25), min(h, ys.max() + 25)
        roi = img_raw[y1:y2, x1:x2]
        overlay = img_display.copy()
        overlay[mask_np > 0] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.4, img_display, 0.6, 0, img_display)
    return img_display, roi


class AnalysisThread(QThread):
    segmentation_ready = pyqtSignal(np.ndarray)
    wiki_ready = pyqtSignal(str, str)
    roi_ready = pyqtSignal(np.ndarray)
    _model = None

    def __init__(self, vs, ds, anime, raw, mode, data):
        super().__init__()
        self.vs, self.ds, self.anime, self.raw, self.mode, self.data = vs, ds, anime, raw, mode, data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        try:
            from ultralytics import FastSAM
            if AnalysisThread._model is None:
                model_full_path = get_resource_path("FastSAM-s.pt")
                AnalysisThread._model = FastSAM(model_full_path)

            img_disp, roi = get_segment_result(AnalysisThread._model, self.raw, self.data, self.mode, self.device)
            self.segmentation_ready.emit(img_disp)
            self.roi_ready.emit(roi)

            check_prompt = "请判断图片中的主体是否属于动漫、二次元或插画风格？如果是，请只回答'YES'；如果不是，请直接给出该物体的中文名称。"
            v_name = self.vs.identify_object(roi, prompt=check_prompt)
            v_text = v_name.strip().upper()

            name = ""
            source_engine = "VISION"

            if "YES" in v_text:
                max_retries = 3
                for i in range(max_retries):
                    try:
                        res = self.anime.identify(roi, prompt="分析主体轮廓及其领域关联。")
                        name = res.get("name")
                        source_engine = "ANIME"
                        if name and name != "未知目标": break
                    except Exception as e:
                        print(f"Anime API 第 {i + 1} 次尝试失败: {e}")
                        if i == max_retries - 1:
                            name = self.vs.identify_object(roi, prompt="这是一个动漫角色，请告诉我它的具体名字。")
                if not name: name = "动漫人物"
            else:
                name = v_name
                source_engine = "VISION"

            wiki = self.ds.get_wiki(name)
            info = f"引擎：{source_engine}\n\n{wiki}"
            self.wiki_ready.emit(name, info)

        except Exception as e:
            self.wiki_ready.emit("错误", str(e))


# =========================
# 🧠 主窗口
# =========================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 百科（终极版）")
        self.resize(1280, 850)

        # ⭐ 设置窗口图标（logo.png 需在打包参数 --add-data 中包含）
        self.setWindowIcon(QIcon(get_resource_path("logo.png")))

        self.statusBar().showMessage("正在初始化引擎...")
        QApplication.processEvents()

        self.voice_mgr = VoiceManager()

        self.vs = VisionEngine(api_key="sk-df077d87d84d486e9c2a9e7964f3959a")
        self.ds = DeepSeekEngine(api_key="sk-52022cb4e05b491fa90576fb72756a74")
        self.anime = AnimeEngine(self.vs)

        self.current_name = ""
        self.current_wiki = ""
        self.mode = SelectMode.POINT
        self.raw = None
        self.thread = None
        self.camera_thread = None
        self.chat_thread = None

        self.init_ui()
        self.statusBar().showMessage("就绪 | 当前模式：点选")

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(F)")

        load_act = QAction("📂 打开图片...", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self.load)
        file_menu.addAction(load_act)

        camera_act = QAction("📷 开启摄像头", self)
        camera_act.setShortcut("Ctrl+K")
        camera_act.triggered.connect(self.toggle_camera)
        file_menu.addAction(camera_act)

        mode_menu = menubar.addMenu("模式(M)")
        mode_group = QActionGroup(self)
        self.act_point = QAction("📍 点选模式", self, checkable=True)
        self.act_point.setChecked(True)
        self.act_point.triggered.connect(lambda: self.set_mode(SelectMode.POINT))
        self.act_box = QAction("🔲 框选模式", self, checkable=True)
        self.act_box.triggered.connect(lambda: self.set_mode(SelectMode.BOX))
        for a in [self.act_point, self.act_box]:
            mode_group.addAction(a)
            mode_menu.addAction(a)

        voice_menu = menubar.addMenu("语音(V)")
        voice_configs = [("男声", "male"), ("女声", "female"), ("油库里", "yukkuri")]
        for text, m in voice_configs:
            act = QAction(text, self)
            act.triggered.connect(lambda _, mode=m: self.voice_mgr.set_mode(mode))
            voice_menu.addAction(act)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.stack = QStackedWidget()
        self.welcome_widget = QWidget()
        welcome_layout = QHBoxLayout(self.welcome_widget)
        welcome_layout.setSpacing(15)

        btn_style = """
            QPushButton {
                background-color: #3b3e4e;
                color: #f8f8f2;
                border: 2px solid #44475a;
                border-radius: 12px;
                font-size: 26px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #50fa7b;
                color: #282a36;
                border: 2px solid #50fa7b;
            }
        """

        self.btn_open_file = QPushButton("📂\n点击载入图片")
        self.btn_open_file.setStyleSheet(btn_style)
        self.btn_open_file.clicked.connect(self.load)
        self.btn_open_camera = QPushButton("📷\n调用摄像头拍照")
        self.btn_open_camera.setStyleSheet(btn_style)
        self.btn_open_camera.clicked.connect(self.toggle_camera)

        welcome_layout.addWidget(self.btn_open_file)
        welcome_layout.addWidget(self.btn_open_camera)

        self.canvas = ImageCanvas()
        self.btn_speak = QPushButton("🔊 朗读百科", self.canvas)
        self.btn_speak.setFixedSize(110, 35)
        self.btn_speak.move(20, 20)
        self.btn_speak.setStyleSheet("background: rgba(255,255,255,180); border-radius:5px; font-weight:bold;")
        self.btn_speak.clicked.connect(self.on_speak_clicked)
        self.btn_speak.hide()

        self.btn_capture = QPushButton("📸 拍摄当前画面", self.canvas)
        self.btn_capture.setFixedSize(140, 40)
        self.btn_capture.move(20, 65)
        self.btn_capture.setStyleSheet("background: #ff5555; color: white; border-radius:5px; font-weight:bold;")
        self.btn_capture.clicked.connect(self.capture_photo)
        self.btn_capture.hide()

        self.stack.addWidget(self.welcome_widget)
        self.stack.addWidget(self.canvas)
        layout.addWidget(self.stack, 3)

        side = QVBoxLayout()
        self.name_label = QLabel("等待...")
        self.name_label.setStyleSheet("font-size:18px; color:#50FA7B; font-weight:bold;")

        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.info_area.setStyleSheet("background: #282a36; color: #f8f8f2; font-family: 'Microsoft YaHei';")

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(260, 160)
        self.preview_label.setStyleSheet("border:1px solid #444; background:#282a36;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ask_label = QLabel("【向百科提问】")
        chat_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("请输入您想追问的内容...")
        self.chat_input.setStyleSheet(
            "height: 35px; border-radius: 5px; padding: 5px; background: #3b3e4e; color: white;")
        self.chat_input.returnPressed.connect(self.on_ask_clicked)

        self.btn_send_ask = QPushButton("🚀")
        self.btn_send_ask.setFixedSize(35, 35)
        self.btn_send_ask.setStyleSheet("background: #50fa7b; color: #282a36; border-radius: 5px; font-weight: bold;")
        self.btn_send_ask.clicked.connect(self.on_ask_clicked)

        chat_layout.addWidget(self.chat_input)
        chat_layout.addWidget(self.btn_send_ask)

        side.addWidget(QLabel("【目标名称】"))
        side.addWidget(self.name_label)
        side.addWidget(QLabel("【详细百科】"))
        side.addWidget(self.info_area, 3)

        side.addWidget(self.ask_label)
        side.addLayout(chat_layout)

        side.addWidget(QLabel("【分析预览】"))
        side.addWidget(self.preview_label)
        layout.addLayout(side, 1)

        self.canvas.area_selected.connect(self.on_box)
        self.canvas.point_clicked.connect(self.on_point)

    # ================= 业务逻辑 =================
    def on_ask_clicked(self):
        text = self.chat_input.text().strip()
        if not text: return
        if not self.current_wiki:
            self.statusBar().showMessage("❌ 请先识别一个目标后再提问")
            return

        self.info_area.append(f"\n🙋‍♂️ **提问**：{text}")
        self.chat_input.clear()
        self.statusBar().showMessage("🤖 AI 正在深入思考...")

        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.terminate()

        self.chat_thread = ChatThread(self.ds, self.current_wiki, text)
        self.chat_thread.answer_ready.connect(self.on_answer_ready)
        self.chat_thread.start()

    def on_answer_ready(self, answer):
        self.info_area.append(f"\n💡 **回答**：{answer}")
        self.info_area.moveCursor(QTextCursor.MoveOperation.End)
        self.statusBar().showMessage("✅ 回答完毕")

    def toggle_camera(self):
        self.stop_analysis()
        self.stack.setCurrentIndex(1)
        self.btn_capture.show()
        self.btn_speak.hide()
        if self.camera_thread and self.camera_thread.isRunning(): return
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.canvas.set_image)
        self.camera_thread.start()

    def capture_photo(self):
        if self.camera_thread:
            self.raw = self.canvas.image.copy()
            self.stop_camera()
            self.btn_capture.hide()
            self.statusBar().showMessage("✅ 拍照成功！")

    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

    def load(self):
        self.stop_camera()
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.raw = cv2.imread(path)
            self.canvas.set_image(self.raw)
            self.stack.setCurrentIndex(1)
            self.btn_capture.hide()

    def set_mode(self, m):
        self.mode = m

    def on_box(self, r):
        if self.mode == SelectMode.BOX: self.start(r)

    def on_point(self, p):
        if self.mode == SelectMode.POINT: self.start(p)

    def stop_analysis(self):
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()

    def start(self, data):
        if self.raw is None: return
        self.stop_camera()
        self.stop_analysis()
        self.thread = AnalysisThread(self.vs, self.ds, self.anime, self.raw, self.mode, data)
        self.thread.segmentation_ready.connect(self.canvas.set_image)
        self.thread.roi_ready.connect(self.show_roi)
        self.thread.wiki_ready.connect(self.show_text)
        self.thread.start()

    def show_roi(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, roi.shape[1], roi.shape[0], roi.shape[1] * 3, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(260, 160, Qt.AspectRatioMode.KeepAspectRatio))

    def show_text(self, n, w):
        self.name_label.setText(n)
        self.info_area.setText(w)
        self.current_name, self.current_wiki = n, w
        self.btn_speak.show()

    def on_speak_clicked(self):
        if self.voice_mgr.is_playing:
            self.stop_voice()
            return
        if self.current_name and self.current_wiki:
            import pygame
            try:
                if pygame.mixer.get_init(): pygame.mixer.music.stop()
            except:
                pass
            parts = self.current_wiki.split('\n\n')
            main_content_parts = [p.strip() for p in parts if not p.strip().startswith(("引擎：", "【候选】"))]
            full_speech = f"识别结果：{self.current_name}。{' '.join(main_content_parts)}"
            self.voice_mgr.speak(full_speech)
            self.btn_speak.setText("⏹️ 停止播放")
            self.btn_speak.setStyleSheet("background: #ff5555; color: white; border-radius:5px; font-weight:bold;")
            self.status_timer = QTimer(self)
            self.status_timer.timeout.connect(self.check_voice_status)
            self.status_timer.start(500)

    def stop_voice(self):
        self.voice_mgr.quit()
        self.voice_mgr = VoiceManager()
        self.btn_speak.setText("🔊 朗读百科")
        self.btn_speak.setStyleSheet("background: rgba(255,255,255,180); border-radius:5px; font-weight:bold;")
        if hasattr(self, 'status_timer'): self.status_timer.stop()

    def check_voice_status(self):
        if not self.voice_mgr.is_playing:
            self.btn_speak.setText("🔊 朗读百科")
            self.btn_speak.setStyleSheet("background: rgba(255,255,255,180); border-radius:5px; font-weight:bold;")
            if hasattr(self, 'status_timer'): self.status_timer.stop()

    def closeEvent(self, event):
        self.stop_camera()
        if hasattr(self, 'voice_mgr'): self.voice_mgr.quit()
        event.accept()


if __name__ == "__main__":
    # ⭐ 声明独立的 App ID，确保任务栏图标显示正确
    try:
        my_appid = 'myteam.encyclopedia.v1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_appid)
    except:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = App()
    win.show()
    sys.exit(app.exec())