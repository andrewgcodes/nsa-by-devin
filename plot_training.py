import matplotlib.pyplot as plt
import numpy as np

# Sample training data points
iterations = [0, 129, 362, 558, 735]  # From training output
losses = [2.5661, 2.5006, 2.4512, 2.4356]  # From training output

plt.figure(figsize=(10, 6))
plt.plot(iterations[:-1], losses, 'b-', label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('NSA Training Progress on Shakespeare Dataset')
plt.grid(True)
plt.legend()
plt.savefig('training_progress.png')
plt.close()
