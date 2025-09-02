import pandas as pd

#A
d1a = pd.read_csv('Lab2 D1A.csv')
d1b = pd.read_csv('Lab2 D1B.csv')

print("Columns in d1a:")
print(d1a.columns)
print("\nColumns in d1b:")
print(d1b.columns)

#B
common_key = 'name'
merged = pd.merge(d1a, d1b, on=common_key, how='inner', suffixes=('', '_dup'))

cols_to_drop = [col for col in merged.columns if col.endswith('_dup')]
merged.drop(columns=cols_to_drop, inplace=True)

print("\nMerged DataFrame shape:", merged.shape)
print("Merged DataFrame columns:")
print(merged.columns)

#C
d1c = pd.read_csv('Lab2 D1C.csv')

comboAC = pd.merge(d1a, d1c, how='inner')

print("\ncombo AC DataFrame shape:", comboAC.shape)
print("\ncombo AC DataFrame columns:")
print(comboAC.columns)
