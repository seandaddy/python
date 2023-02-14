import random

data = [random.randint(1,100) for _ in range(100)]
print(data)

print(f"Sum:{sum(data)}, N:{len(data)}")
print(f"Average:{sum(data)/len(data)}")
