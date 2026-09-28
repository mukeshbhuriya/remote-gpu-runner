import time
import os

print("Starting test job...")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

for i in range(5):
    print(f"Step {i}")
    time.sleep(1)

print("Test job completed successfully.")
