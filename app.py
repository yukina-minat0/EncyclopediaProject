import os
import sys
import cv2
import torch
import numpy as np
from enum import Enum

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# 假设你的引擎文件都在同级目录下
from vision import VisionEngine
from deepseek_engine import DeepSeekEngine
from anime_engine import AnimeEngine

# 限制线程数，防止 CPU 抢占导致 UI 卡顿
torch.set_num_threads(1)


class SelectMode(Enum):
    BOX = 0
    POINT = 1


# =========================
# 🎨 自绘画布
# =========================
class ImageCanvas(QWidget):
    area_selected = pyqtSignal(QRect)
    point_clicked = pyqtSignal(QPoint)

    def __init__(self):
        super().__init__()
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

        if self.image is None:
            return

        h, w = self.image.shape[:2]
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)

        scale_w = self.width() / w
        scale_h = self.height() / h
        self.scale = min(scale_w, scale_h)

        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        self.offset = QPoint(
            (self.width() - new_w) // 2,
            (self.height() - new_h) // 2
        )

        painter.drawImage(QRect(self.offset, QSize(new_w, new_h)), qimg)

        if self.begin and self.end:
            painter.setPen(QPen(QColor(0, 255, 127), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.begin, self.end))

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

            # 判定为点击还是拖拽
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
        x = int((r.x() - self.offset.x()) / self.scale)
        y = int((r.y() - self.offset.y()) / self.scale)
        w = int(r.width() / self.scale)
        h = int(r.height() / self.scale)
        return QRect(x, y, w, h)


# =========================
# 🔥 分析线程
# =========================
class AnalysisThread(QThread):
    segmentation_ready = pyqtSignal(np.ndarray)
    wiki_ready = pyqtSignal(str, str)
    roi_ready = pyqtSignal(np.ndarray)

    _model = None  # 静态变量共享模型

    def __init__(self, vs, ds, anime, raw, mode, data, device_id=0):
        super().__init__()
        self.vs = vs
        self.ds = ds
        self.anime = anime
        self.raw = raw
        self.mode = mode
        self.data = data
        # 显存安全检查：优先使用 CUDA，报错时可手动切回 cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        try:
            from ultralytics import FastSAM
            if AnalysisThread._model is None:
                AnalysisThread._model = FastSAM("FastSAM-s.pt")

            model = AnalysisThread._model
            img_display = self.raw.copy()
            h, w = self.raw.shape[:2]

            if self.mode == SelectMode.BOX:
                r = self.data
                x1, y1 = max(0, r.x()), max(0, r.y())
                x2, y2 = min(w, r.right()), min(h, r.bottom())

                # 框选模式下的 ROI 扩展逻辑
                expansion_factor = 0.3
                w_box, h_box = x2 - x1, y2 - y1
                x1_ex = max(0, int(x1 - w_box * expansion_factor))
                y1_ex = max(0, int(y1 - h_box * expansion_factor))
                x2_ex = min(w, int(x2 + w_box * expansion_factor))
                y2_ex = min(h, int(y2 + h_box * expansion_factor))
                roi = self.raw[y1_ex:y2_ex, x1_ex:x2_ex]

            else:
                # 点选模式逻辑
                x, y = self.data.x(), self.data.y()
                x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)

                results = model.predict(
                    source=self.raw,
                    points=[[x, y]],
                    labels=[1],
                    device=self.device,
                    retina_masks=True,
                    verbose=False
                )

                masks = results[0].masks.data
                if masks is None or len(masks) == 0:
                    raise Exception("未能分割出有效目标")

                mask_areas = [(m.sum(), m) for m in masks]
                mask_areas.sort(reverse=True, key=lambda x: x[0])

                best_mask = mask_areas[0][1]
                second_best = mask_areas[1][1] if len(mask_areas) > 1 else None
                # 合并掩码以增加识别准确度
                combined_mask = best_mask | second_best if second_best is not None else best_mask

                mask_np = combined_mask.cpu().numpy()
                ys, xs = np.where(mask_np > 0)

                # 裁剪带边距的 ROI
                x1, x2 = max(0, xs.min() - 25), min(w, xs.max() + 25)
                y1, y2 = max(0, ys.min() - 25), min(h, ys.max() + 25)
                roi = self.raw[y1:y2, x1:x2]

                # 绘制分割反馈效果
                overlay = img_display.copy()
                overlay[mask_np > 0] = [0, 0, 255]
                cv2.addWeighted(overlay, 0.4, img_display, 0.6, 0, img_display)

            self.segmentation_ready.emit(img_display)
            self.roi_ready.emit(roi)

            # 💡 识别与百科逻辑
            prompt = "如果图片背景为纯黑或有明显抠图痕迹，请忽略边缘干扰，重点分析主体轮廓及其与特定领域（如动漫/工业）的关联。"
            result = self.anime.identify(roi, prompt=prompt)

            name = result.get("name", "未知目标")
            candidates = result.get("candidates", [])
            source_engine = result.get("source", "none")

            wiki_content = self.ds.get_wiki(name)
            extra_info = "\n\n【候选列表】\n" + "\n".join(candidates) if candidates else ""
            final_text = f"识别引擎：{source_engine.upper()}\n\n{wiki_content}{extra_info}"

            self.wiki_ready.emit(name, final_text)

        except Exception as e:
            self.wiki_ready.emit("分析中断", f"详细错误: {str(e)}")


# =========================
# 🧠 主窗口
# =========================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 百科（终极版）")
        self.resize(1200, 800)

        # 引擎初始化（请确保 API Key 有效）
        self.vs = VisionEngine(api_key="sk-df077d87d84d486e9c2a9e7964f3959a")
        self.ds = DeepSeekEngine(api_key="sk-52022cb4e05b491fa90576fb72756a74")
        self.anime = AnimeEngine(self.vs)

        self.mode = SelectMode.POINT
        self.raw = None
        self.thread = None

        self.init_ui()
        self.statusBar().showMessage("准备就绪 | 当前模式：点选")

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(F)")

        load_act = QAction("打开图片...", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self.load)
        file_menu.addAction(load_act)

        exit_act = QAction("退出(E)", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        mode_menu = menubar.addMenu("操作模式(M)")
        mode_group = QActionGroup(self)
        self.act_point = QAction("📍 点选模式", self, checkable=True)
        self.act_point.setChecked(True)
        self.act_point.triggered.connect(lambda: self.set_mode(SelectMode.POINT))

        self.act_box = QAction("🔲 框选模式", self, checkable=True)
        self.act_box.triggered.connect(lambda: self.set_mode(SelectMode.BOX))

        mode_group.addAction(self.act_point)
        mode_group.addAction(self.act_box)
        mode_menu.addAction(self.act_point)
        mode_menu.addAction(self.act_box)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧：引导页与画布
        left_layout = QVBoxLayout()
        self.stack = QStackedWidget()

        self.btn_select_first = QPushButton("📂 点击此处选择图片开始分析")
        self.btn_select_first.setStyleSheet("""
            QPushButton {
                background-color: #282a36;
                color: #6272a4;
                font-size: 20px;
                border: 2px dashed #44475a;
                border-radius: 15px;
            }
            QPushButton:hover { background-color: #383a59; color: #f8f8f2; }
        """)
        self.btn_select_first.clicked.connect(self.load)

        self.canvas = ImageCanvas()
        self.stack.addWidget(self.btn_select_first)
        self.stack.addWidget(self.canvas)
        left_layout.addWidget(self.stack)
        layout.addLayout(left_layout, 3)

        # 右侧：展示面板
        side_layout = QVBoxLayout()
        self.name_label = QLabel("等待操作...")
        self.name_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #50FA7B;")

        self.info_area = QLabel("请载入图片并点击目标进行分析。")
        self.info_area.setWordWrap(True)
        self.info_area.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.info_area)

        self.preview_label = QLabel("局部预览")
        self.preview_label.setFixedSize(260, 160)
        self.preview_label.setStyleSheet("border: 1px solid #444; background: #282a36;")

        side_layout.addWidget(QLabel("【目标名称】"))
        side_layout.addWidget(self.name_label)
        side_layout.addWidget(QLabel("【百科详情】"))
        side_layout.addWidget(scroll)
        side_layout.addWidget(QLabel("【目标预览】"))
        side_layout.addWidget(self.preview_label)

        layout.addLayout(side_layout, 1)

        self.canvas.area_selected.connect(self.on_box)
        self.canvas.point_clicked.connect(self.on_point)

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.raw = cv2.imread(path)
            if self.raw is not None:
                self.canvas.set_image(self.raw)
                self.stack.setCurrentIndex(1)
                self.statusBar().showMessage(f"已加载: {os.path.basename(path)}")

    def set_mode(self, m):
        self.mode = m
        self.statusBar().showMessage(f"当前模式：{'点选' if m == SelectMode.POINT else '框选'}")

    def on_box(self, rect):
        if self.mode == SelectMode.BOX: self.start(rect)

    def on_point(self, p):
        if self.mode == SelectMode.POINT: self.start(p)

    def start(self, data):
        if self.raw is None: return

        # 线程安全：如果已有线程在跑，先停止它
        if self.thread and self.thread.isRunning():
            self.thread.terminate()  # 强制停止以响应新点击
            self.thread.wait()

        self.statusBar().showMessage("AI 正在分析中...")
        # 💡 正确传递参数给构造函数
        self.thread = AnalysisThread(self.vs, self.ds, self.anime, self.raw, self.mode, data)
        self.thread.segmentation_ready.connect(self.canvas.set_image)
        self.thread.roi_ready.connect(self.show_roi)
        self.thread.wiki_ready.connect(self.show_text)
        self.thread.finished.connect(lambda: self.statusBar().showMessage("分析完成"))
        self.thread.start()

    def show_roi(self, roi):
        if roi is None or roi.size == 0: return
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        h, w = roi.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def show_text(self, n, w):
        self.name_label.setText(n)
        self.info_area.setText(w)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = App()
    win.show()
    sys.exit(app.exec())