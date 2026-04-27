import requests
import base64
import cv2
import time

class AnimeEngine:
    def __init__(self, vision_engine=None):
        self.url = "https://dio.jite.me/api/recognize"
        self.vision = vision_engine
        self.similarity_threshold = 0.6
        self.top_k = 3
        self.retry_count = 3  # 设置最大重试次数

    def _preprocess(self, img):
        """将 OpenCV 图像转为可用于网络传输的 JPG 字节流"""
        if img is None or img.size == 0:
            return None

        h, w = img.shape[:2]

        # 太小直接放大，提高 API 识别率
        if min(h, w) < 128:
            scale = 128 / min(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # 增强图像：增强对比度等
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # 增强对比度

        # 编码为 jpg，获取编码状态和内存 buffer
        success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if not success:
            return None

        # 返回真实的纯字节流（bytes）
        return buffer.tobytes()

    def _send_request(self, img_bytes):
        """请求 Anime API，支持重试机制"""
        for attempt in range(self.retry_count):
            try:
                resp = requests.post(
                    self.url,
                    files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                    data={"use_correction": "1"},  # 启用全局修正
                    timeout=10
                )

                if resp.status_code == 200:
                    return resp.json()
                else:
                    print(f"API 错误，状态码: {resp.status_code}, 尝试第 {attempt+1} 次重试...")
                    time.sleep(2)

            except Exception as e:
                print(f"Anime API 请求失败: {e}, 尝试第 {attempt+1} 次重试...")
                time.sleep(2)

        return None

    def identify(self, img, prompt=None):  # ⭐ 核心修正 1：增加 prompt 参数接收
        """
        返回：
        {
            "name": 主识别结果,
            "candidates": [候选列表],
            "source": "anime" 或 "vision"
        }
        """
        img_bytes = self._preprocess(img)

        if img_bytes is None:
            return {
                "name": "无效图像",
                "candidates": [],
                "source": "none"
            }

        try:
            # 1. 首先尝试专用 Anime API (此 API 通常不接受自定义 prompt，它是固定算法)
            data = self._send_request(img_bytes)

            if data and "faces" in data and data["faces"]:
                results = data["faces"][:self.top_k]

                candidates = []
                for r in results:
                    sim = r.get("score", 0)
                    if sim < self.similarity_threshold:
                        continue

                    name = r.get("name", "未知")
                    anime = r.get("anime", "")
                    candidates.append(f"{name}（{anime}） {sim:.2f}")

                if candidates:
                    return {
                        "name": candidates[0],
                        "candidates": candidates,
                        "source": "anime"
                    }

        except Exception as e:
            print(f"AnimeEngine 识别异常: {e}")

        # ===== 2. 兜底 Vision (这里是真正需要 prompt 的地方) =====
        if self.vision:
            try:
                # ⭐ 核心修正 2：将 prompt 传递给底层的 Vision 引擎
                # 注意：请确保你的 vision_engine.py 中的 identify_object 方法也支持 prompt 参数
                name = self.vision.identify_object(img, prompt=prompt)
                return {
                    "name": name,
                    "candidates": [],
                    "source": "vision"
                }
            except Exception as e:
                print(f"Vision 兜底失败: {e}")

        return {
            "name": "未识别出角色",
            "candidates": [],
            "source": "none"
        }