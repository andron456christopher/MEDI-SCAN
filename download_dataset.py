import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")

print("Dataset downloaded at:", path)

# Show files inside folder
print("\nFiles:")
for file in os.listdir(path):
    print(file)