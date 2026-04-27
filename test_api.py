import time
# 确保你的 voice_manager.py 在同一目录下
from voice_manager import VoiceManager


def test_voice_system():
    # 1. 初始化引擎
    print("正在启动语音系统测试...")
    vm = VoiceManager()

    # 2. 切换到油库里模式
    vm.set_mode("yukkuri")

    # 3. 准备测试文本
    # 包含：前言废话（应被跳过）、Markdown标题、中文、英文缩写、英文单词、数字
    test_text = """
由于你查询了相关信息，以下是关于 AI 技术的百科介绍：

### 百科正文：人工智能
人工智能（Artificial Intelligence），简称 AI。
它在 2024 年的发展非常迅速。
常用的模型包括 GPT-4 和 Claude 3.5 版本。
你可以说 Hello 给机器人，它会回复你。
0.618 是黄金分割率。
    """

    print("\n--- 开始朗读测试 ---")
    print("预期行为：")
    print("1. 跳过‘由于你查询...’段落")
    print("2. AI 读作 'ee-ai'")
    print("3. Hello 读作 'はろー' (基于 wanakana 拟音)")
    print("4. 0.618 读作 'zero ten roku ichi hachi'")

    # 4. 执行朗读
    vm.speak(test_text)

    # 等待朗读完成（因为 speak 是异步线程）
    try:
        while vm.is_playing:
            time.sleep(0.5)
    except KeyboardInterrupt:
        vm.quit()

    print("\n--- 测试完成 ---")
    vm.quit()


if __name__ == "__main__":
    test_voice_system()