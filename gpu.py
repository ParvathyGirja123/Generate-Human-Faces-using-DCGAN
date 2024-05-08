import tensorflow as tf

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU available: Yes")
    # Set TensorFlow to use GPU
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("GPU available: No")


import tensorflow as tf

# Explicitly specify to use CPU
tf.config.set_visible_devices([], 'GPU')

# Check if CPU is available
if tf.config.list_physical_devices('CPU'):
    print("CPU available: Yes")
else:
    print("CPU available: No")
    
if tf.config.list_physical_devices('GPU'):
    # GPU is available
    device = 'GPU'
else:
    # No GPU available, fallback to CPU
    device = 'CPU'