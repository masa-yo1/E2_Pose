import tensorflow as tf

# 利用可能なGPUデバイスのリストを表示
print("利用可能なGPUデバイス:")
for device in tf.config.list_physical_devices('GPU'):
    print(device)

# TensorFlowがGPUを利用するかどうかを確認
print("GPUが利用可能か:", tf.test.is_gpu_available())
