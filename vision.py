import base64
import cv2
from openai import OpenAI


class VisionEngine:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = "qwen3-vl-flash"

    def identify_object(self, numpy_img, prompt=None):  # ⭐ 修正：增加 prompt 参数接收
        if numpy_img is None or numpy_img.size == 0:
            return "无效图像"

        # 编码图像
        _, buffer = cv2.imencode('.jpg', numpy_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        answer_content = ""

        # ⭐ 核心升级：内置专家 Prompt
        base_prompt = """
你是一个精通多领域的视觉识别专家。请分析这张图片，识别出物体的【唯一精确名称】：

1. **领域识别优先：**
   - **动漫/游戏/二次元：** 请基于角色发型、瞳色、服装或标志性饰品识别出【角色名+作品来源】。严禁输出“动漫女孩”等泛称。
   - **工业/机械零件：** 观察表面纹理、加工痕迹（如滚花、螺纹、台阶）。识别具体的工程术语，如“阶梯轴”、“滚针轴承”、“六角槽栓”等。
   - **专业设备/电子产品：** 识别具体的【品牌 + 型号】，如“罗技 G502”而非“鼠标”。

2. **识别约束：**
   - 如果物体具有品牌Logo、特定文字，请作为核心识别依据。
   - 严禁输出任何解释、推测或前导词（如“图片中显示了...”）。
   - 如果无法完全确定，请输出与其视觉特征最契合的【特定称呼】。

3. **输出要求：**
   - 只输出名称，不带标点符号。
"""
        # ⭐ 逻辑合并：如果有外部传入的针对性 Prompt（如关于抠图的指令），将其附加在后面
        final_prompt = base_prompt
        if prompt:
            final_prompt += f"\n【补充指令】：{prompt}"

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": final_prompt},
                        ],
                    },
                ],
                # 开启思考链路，有助于处理复杂物体
                extra_body={"enable_thinking": True},
                stream=True
            )

            for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    answer_content += delta.content

            result = answer_content.strip()

            # ⭐ 兜底清理：移除常见的引导性前缀
            prefixes_to_remove = ["名称：", "目标是：", "物体：", "结果：", "识别结果："]
            for p in prefixes_to_remove:
                if result.startswith(p):
                    result = result.replace(p, "", 1)

            # ⭐ 长度限制，确保 UI 显示美观
            if len(result) > 50:
                result = result[:50]

            return result if result else "未知目标"

        except Exception as e:
            print(f"【视觉识别异常】: {e}")
            return "识别失败"