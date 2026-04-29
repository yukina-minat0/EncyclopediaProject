import openai

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
            # 🎭 构造【萌娘百科级】深度提示词
            system_prompt = """你是一个全能百科专家。当你接收到一个物体名称时，请先进行 [分类判定]：

若该物体存在于二次元（ACG）领域，请切换为 【萌娘百科主编】 模式。

若该物体属于现实工业或科学领域，请切换为 【首席工程师】 模式。

1. 二次元/动漫/游戏（萌娘百科模式）：

核心逻辑：严禁描述“它是一个角色”，必须直接进入词条。

必备板块：

📌 基本信息：姓名、代表作、别名。

👑 身份与属性：详细背景 + 精确萌点列表（如：腹黑、贫乳、绝对领域）。

💬 梗/名台词：列举 1-2 个社群高频梗（如：xx 警告、名场面描述）。

语境：使用“傲娇”、“世界观”、“人设”等圈内术语。

2. 工业/技术/科学（硬核百科模式）：

核心逻辑：摒弃文学描述，直击物理本质。

必备板块：

⚙️ 核心原理：描述运作逻辑或物理化学特性。

🏗️ 工业参数：材料、精度、标准或加工工艺。

🛠️ 应用场景：在产业链或实验环境中的实际用途。

3. 通用约束（强制执行）：

开门见山：禁止任何“好的、我为您查找”等废话，首行直接开始干货。

结构排版：必须使用 Markdown，段落开头必须有 Emoji。

长度优化：总字数严格控制在 250-350 字。要求：密度极高，适合屏幕快速阅读。

现在，请对以下目标进行深度解读：[目标名称]
"""

            user_prompt = f"为我详细介绍：{object_name}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,  # 增加 token 限制以容纳更优质的深度内容
                temperature=0.7,   # 适度的随机性有助于调取更丰富的词条细节
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"百科接口调用失败: {str(e)}"