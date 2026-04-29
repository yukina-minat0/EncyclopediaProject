from modelscope.hub.file_download import model_file_download

# 使用魔搭社区官方镜像的 whisper-small 路径
# 这是一个更标准的 ID 格式
try:
    model_dir = model_file_download(
        model_id='damo/speech_whisper-small_generation_multi-lingual',
        file_path='whisper-small.pt'
    )
    print(f"模型下载完成，路径在：{model_dir}")
except Exception as e:
    print(f"下载失败，错误信息：{e}")
