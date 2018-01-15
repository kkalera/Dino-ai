import numpy as np

a = np.array([["value1", "value2", 3, "value4", "value5"],
             ["value6", "value7", -10, "value8", "value9"],
             ["value10", "value11", 31, "value12", "value13"],
             ["value14", "value15", 5, "value16", "value17"],
             ["value18", "value19", 3, "value20", "value21"]])

print("Default")
print(a)

a = a[a[:, 2].astype(np.int).argsort()]

print()
print("Sorted:")
print(a)
