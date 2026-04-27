import os
import torch

# --- 核心：针对 RTX 5060 (Blackwell架构) 的兼容性黑科技 ---
# 强制开启延迟加载，防止 PyTorch 因为不认识新显卡架构而直接崩溃
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

from ultralytics import FastSAM


def final_test():
    print("--- 正在启动 RTX 5060 专项测试 ---")

    # 1. 显卡状态确认
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    if cuda_available:
        print(f"检测到顶级显卡: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("警告: 显卡未被识别，请检查驱动是否更新到 2026 年版本")
        device = 'cpu'

    # 2. 绝对路径定位（彻底解决 FileNotFoundError）
    # 获取当前脚本所在文件夹的绝对路径
    base_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_path, 'test.jpg')

    if not os.path.exists(img_path):
        print(f"❌ 依然找不到图片！")
        print(f"请确认你的图片文件是否准确命名为 'test.jpg' 并放在：\n{base_path}")
        return

    # 3. 加载模型并推理
    print(f"正在使用 {device} 加载 FastSAM 模型...")
    model = FastSAM('FastSAM-s.pt')

    # 尝试点选坐标 (500, 500) 进行分割
    results = model.predict(
        source=img_path,
        device=device,
        points=[[500, 500]],
        labels=[1],
        retina_masks=True,
        imgsz=1024
    )

    # 4. 保存结果
    output_path = os.path.join(base_path, 'result_5060_final.jpg')
    results[0].save(filename=output_path)
    print(f"✅ 成功生成结果图！请在文件夹中查看: {output_path}")


if __name__ == "__main__":
    final_test()