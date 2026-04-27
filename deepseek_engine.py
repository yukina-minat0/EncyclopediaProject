import openai  # 💡 记得执行 pip install openai


class DeepSeekEngine:
    def __init__(self, api_key):
        # 💡 指向官方服务器地址
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        # 💡 使用 V3 模型 (deepseek-chat)
        self.model = "deepseek-chat"

    def get_wiki(self, object_name):
        try:
            # 🎭 构造领域感知的深度提示词
            system_prompt = """你是一个全能百科专家。请根据用户提供的物体名称进行分类深度解读：

1. **领域识别与深度分配：**
   - **二次元/动漫/游戏领域**：如果你判断该目标属于此类，请展现“资深宅”的专业度。详细说明：作品出处、角色核心设定（性格/萌点）、标志性能力/武器、在剧情中的地位，以及在社群中的流行梗。
   - **工业/技术/科学领域**：展现“工程师”的严谨。详细说明：工作原理、工业用途、核心参数标准及相关技术背景。
   - **日常生活/通用领域**：简明扼要，说明主要用途和常识即可。

2. **核心原则：**
   - **禁止跨领域干扰**：如果目标不是二次元，严禁使用动漫术语或进行二次元类比。
   - **精确定位**：直接给出最精确的称呼，拒绝宽泛词汇。
   - **排版规范**：使用分段排版，每段开头使用适当的 Emoji 增加可读性。

直接从名词的核心定义或 ### 标题开始输出内容。 保持排版简洁，适合语音朗读。
"""

            user_prompt = f"请为我介绍该目标的百科知识：{object_name}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,  # 允许生成更详细的内容
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"百科接口调用失败: {str(e)}"

# ==========================================
# 💡 效果示例（逻辑演示）：
# ==========================================
# 输入 "博丽灵梦" -> 输出：详细的作品出处、能力设定、相关同人梗。
# 输入 "阶梯轴" -> 输出：机械结构原理、加工工艺、配合公差等工业知识。
# 输入 "苹果" -> 输出：植物学分类、主要用途（简洁版）。