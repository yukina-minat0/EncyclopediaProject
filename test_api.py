import time
from voice_manager import VoiceManager


def test_all_modes():
    print("🚀 开始语音引擎专项测试...")
    vm = VoiceManager()

    # 测试文本
    test_text = "这是一段测试文本。现在的感觉怎么样呢？"

    # 定义测试序列
    modes = [
        ("女声测试", "female"),
        ("男声测试", "male"),
        ("油库里模式测试", "yukkuri")
    ]

    for label, mode_name in modes:
        print(f"\n正在切换到模式: {label} ({mode_name})")
        vm.set_mode(mode_name)

        # 稍微等一下，确保模式切换指令执行完
        time.sleep(1)

        print(f"🎙️ 正在播放: {label}...")
        vm.speak(f"当前是{label}。{test_text}")

        # 等待播放完成（根据你的 VoiceManager 逻辑，这里可能需要手动根据 is_playing 判断）
        timeout = 10
        start_time = time.time()
        while vm.is_playing and (time.time() - start_time < timeout):
            time.sleep(0.5)

    print("\n✅ 所有模式测试完毕，请检查是否有哪种声音没响。")
    vm.quit()


if __name__ == "__main__":
    test_all_modes()