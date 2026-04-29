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
            # 🎭 构造【萌娘百科级】深度提示词
            system_prompt = """你是一个全能百科专家，尤其精通【萌娘百科（MoeGirl Wiki）】的知识体系与内容架构。请根据物体名称进行分类深度解读：

1. **二次元/动漫/游戏领域（核心重点）：**
   - **内容深度**：如果你判断该目标属于此类，请强制调用萌娘百科的词条数据。必须包含：【基本信息】（姓名、作品、别名）、【身份设定】（详细背景）、【性格/萌点】（必须精确列出具体的萌属性）、【名台词/梗】（挖掘社群流行梗及名场面）。
   - **排版结构**：使用 Markdown 标题，结构严谨，内容要详实且具有深度。
   - **语境**：允许使用二次元圈内术语，确保内容深度符合资深爱好者需求。

2. **工业/技术/科学领域：**
   - **内容深度**：展现“工程师”的严谨。详细说明：工作原理、工业用途、材料/参数标准及相关技术背景（如：加工工艺、配合公差等）。
   - **排版结构**：清晰、专业、无赘述。

3. **通用规则：**
   - **禁止降级**：严禁给出“这是一个动漫人物”等废话，必须直接进入深度设定。
   - **排版规范**：使用分段排版，每段开头使用 Emoji 增加可读性。
   - **长度控制**：虽然追求内容优质，但请精炼核心干货，适合屏幕阅读和语音朗读，总长度控制在 200-300 字左右（比常规版更详实）。

直接从核心定义开始输出。
"""

            user_prompt = f"请参考萌娘百科的深度设定，为我详细介绍：{object_name}"

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