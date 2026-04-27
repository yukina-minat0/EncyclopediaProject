import os
import sys
import cv2
import torch
import numpy as np
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

class SelectMode(Enum):
    BOX = 0
    POINT = 1


class SelectMode(Enum):
    BOX = 0
    POINT = 1


# =========================
# 🎨 自绘画布 (保持原样)
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
# 🔥 分析线程核心逻辑 (抽离函数)
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
                AnalysisThread._model = FastSAM("FastSAM-s.pt")

            img_disp, roi = get_segment_result(AnalysisThread._model, self.raw, self.data, self.mode, self.device)
            self.segmentation_ready.emit(img_disp)
            self.roi_ready.emit(roi)

            res = self.anime.identify(roi, prompt="分析主体轮廓及其领域关联。")
            name = res.get("name", "未知目标")
            wiki = self.ds.get_wiki(name)

            info = f"引擎：{res.get('source', 'none').upper()}\n\n{wiki}"
            if res.get("candidates"):
                info += "\n\n【候选】\n" + "\n".join(res["candidates"])

            self.wiki_ready.emit(name, info)
        except Exception as e:
            self.wiki_ready.emit("错误", str(e))


# =========================
# 🧠 主窗口 (整合版)
# =========================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 百科（终极版）")
        self.resize(1200, 800)

        # 1. 引擎与逻辑初始化 (单例模式)
        # 💡 在状态栏给用户反馈，因为启动 Chrome 需要时间
        self.statusBar().showMessage("正在初始化语音引擎 (Loading Chrome)...")
        QApplication.processEvents()  # 强制刷新界面显示状态栏

        self.voice_mgr = VoiceManager()
        self.vs = VisionEngine(api_key="sk-df077d87d84d486e9c2a9e7964f3959a")
        self.ds = DeepSeekEngine(api_key="sk-52022cb4e05b491fa90576fb72756a74")
        self.anime = AnimeEngine(self.vs)

        self.current_name = ""
        self.current_wiki = ""
        self.mode = SelectMode.POINT
        self.raw = None
        self.thread = None

        # 2. 构建界面
        self.init_ui()
        self.statusBar().showMessage("准备就绪 | 当前模式：点选")

    def init_ui(self):
        # --- 菜单栏 ---
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(F)")
        load_act = QAction("打开图片...", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self.load)
        file_menu.addAction(load_act)

        # 模式菜单
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

        # 语音菜单
        voice_menu = menubar.addMenu("语音(V)")
        voice_configs = [("男声", "male"), ("女声", "female"), ("油库里", "yukkuri")]
        for text, m in voice_configs:
            act = QAction(text, self)
            act.triggered.connect(lambda _, mode=m: self.voice_mgr.set_mode(mode))
            voice_menu.addAction(act)

        # --- 主布局 ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧：画布堆栈
        self.stack = QStackedWidget()
        self.btn_select_first = QPushButton("📂 点击载入图片")
        self.btn_select_first.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.btn_select_first.clicked.connect(self.load)

        # 假设 ImageCanvas 类已在外部定义
        self.canvas = ImageCanvas()

        # 💡 朗读按钮 (放置在画布之上)
        self.btn_speak = QPushButton("🔊 朗读百科", self.canvas)
        self.btn_speak.setFixedSize(100, 35)
        self.btn_speak.move(20, 20)
        self.btn_speak.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 180); 
                border-radius: 5px; 
                font-weight: bold;
                border: 1px solid #ccc;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 255);
                border: 1px solid #50FA7B;
            }
        """)
        self.btn_speak.clicked.connect(self.on_speak_clicked)
        self.btn_speak.hide()

        self.stack.addWidget(self.btn_select_first)
        self.stack.addWidget(self.canvas)
        layout.addWidget(self.stack, 3)

        # 右侧：信息面板
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

        side.addWidget(QLabel("【目标名称】"))
        side.addWidget(self.name_label)
        side.addWidget(QLabel("【详细百科】"))
        side.addWidget(self.info_area)
        side.addWidget(QLabel("【分析预览】"))
        side.addWidget(self.preview_label)
        layout.addLayout(side, 1)

        # 信号连接
        self.canvas.area_selected.connect(self.on_box)
        self.canvas.point_clicked.connect(self.on_point)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.raw = cv2.imread(path)
            self.canvas.set_image(self.raw)
            self.stack.setCurrentIndex(1)

    def set_mode(self, m):
        self.mode = m
        self.statusBar().showMessage(f"当前模式：{'点选' if m == SelectMode.POINT else '框选'}")

    def on_box(self, r):
        if self.mode == SelectMode.BOX: self.start(r)

    def on_point(self, p):
        if self.mode == SelectMode.POINT: self.start(p)

    def start(self, data):
        if self.raw is None: return
        # 如果当前有正在运行的分析线程，先停止它
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()

        self.thread = AnalysisThread(self.vs, self.ds, self.anime, self.raw, self.mode, data)
        self.thread.segmentation_ready.connect(self.canvas.set_image)
        self.thread.roi_ready.connect(self.show_roi)
        self.thread.wiki_ready.connect(self.show_text)
        self.thread.start()

    def show_roi(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        h, w = roi.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(260, 160, Qt.AspectRatioMode.KeepAspectRatio))

    def show_text(self, n, w):
        self.name_label.setText(n)
        self.info_area.setText(w)
        self.current_name, self.current_wiki = n, w
        self.btn_speak.show()

    def on_speak_clicked(self):
        """朗读正文全文，既然 Prompt 已经优化过长度"""
        if self.voice_mgr.is_playing:
            self.statusBar().showMessage("🕒 语音引擎正在忙碌...")
            return

        if self.current_name and self.current_wiki:
            # 1. 停止当前可能存在的残留播放
            import pygame
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except:
                pass

            # 2. 解析文本：跳过“引擎：XXX”这种元数据标注
            # 假设你的格式依然是：引擎信息 \n\n 百科内容 \n\n 候选
            parts = self.current_wiki.split('\n\n')

            # 找到真正的百科描述部分
            # 我们排除掉第一行（引擎名）和包含“【候选】”的部分
            main_content_parts = []
            for p in parts:
                p = p.strip()
                if not p.startswith("引擎：") and not p.startswith("【候选】"):
                    main_content_parts.append(p)

            # 合并剩下的正文
            full_content = " ".join(main_content_parts)

            # 如果正文解析失败，至少读个名字
            if not full_content.strip():
                full_content = self.current_name

            # 3. 拼接最终要读的话
            # 不再使用 .split('。')[0]，直接读全文
            full_speech = f"识别结果：{self.current_name}。{full_content}"

            self.statusBar().showMessage(f"🎙️ 正在播放：{self.current_name}")
            self.voice_mgr.speak(full_speech)

    def closeEvent(self, event):
        """窗口关闭时彻底杀死后台驱动，防止残留进程"""
        print("正在关闭应用，释放资源...")
        self.statusBar().showMessage("正在清理后台进程...")
        if hasattr(self, 'voice_mgr'):
            self.voice_mgr.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = App()
    win.show()
    sys.exit(app.exec())