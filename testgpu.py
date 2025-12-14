import tensorflow as tf
import numpy as np
import time

print("=" * 80)
print("🎮 GPU/CPU TEST")
print("=" * 80)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n✅ GPUs Found: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("   ❌ No GPU detected - using CPU")

# Check CPU
cpus = tf.config.list_physical_devices('CPU')
print(f"\n✅ CPUs Found: {len(cpus)}")

# Test with simple computation
print("\n" + "=" * 80)
print("⚡ PERFORMANCE TEST")
print("=" * 80)

# Create test data
size = 10000
x = tf.random.normal((size, size))
y = tf.random.normal((size, size))

print(f"\nTesting matrix multiplication: {size}x{size}")

# Warm up
_ = tf.matmul(x, y)

# Time it
start = time.time()
result = tf.matmul(x, y)
end = time.time()

time_ms = (end - start) * 1000
print(f"⏱️  Time: {time_ms:.2f} ms")

if time_ms < 500:
    print("🟢 GPU is working! (Fast)")
elif time_ms < 2000:
    print("🟡 CPU is fast enough")
else:
    print("🔴 Slow - check setup")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)