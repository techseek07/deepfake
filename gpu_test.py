from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()
for d in devices:
    if d.device_type == 'GPU':
        print(f"\n🔍 Detected GPU: {d.physical_device_desc}")
