# %%

import random

# %%

lst = [random.randint(1, 1000) for i in range(10)]
reps = [random.randint(1, 15) for i in range(10)]

result = []

for i, j in zip(lst, reps):
    result.extend([i] * j)

# %%

print(list(zip(lst, reps)))
print(result)


# Заведём два списка: в первом будем хранить список уникальных
# \элементов, а во втором - соответствующие количества повторений

# %%

unique = list(set(result))
counts = []

for num in unique:
    counts.append(result.count(num))

print(counts)


# %%
print(reps)
