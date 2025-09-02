#TASK 02
import pandas as pd
import numpy as np

data_a = pd.read_csv('Lab2 D1A.csv')
data_b = pd.read_csv('Lab2 D1B.csv')
data_c = pd.read_csv('Lab2 D1C.csv')

np.random.seed(42)

custom_data = pd.DataFrame({
    'fid': np.arange(1, 1001),
    'location_name': [f'Loc_{i}' for i in range(1, 1001)],
    'size_category': np.random.choice(['small', 'medium', 'large'], 1000),
    'direction_facing': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'work_timing': np.random.choice(['full_time', 'part_time'], 1000),
    'custom_cat': np.random.choice(['A', 'B', 'C'], 1000),
    'custom_score': np.random.rand(1000) * 100
})

print("custom_data sample:")
print(custom_data.head())

merged_ab = pd.merge(custom_data, data_a, left_on='location_name', right_on='name', how='inner')
merged_abc = pd.merge(merged_ab, data_b, left_on='location_name', right_on='name', how='inner', suffixes=('', '_dup'))

cols_to_drop = [col for col in merged_abc.columns if col.endswith('_dup')]
merged_abc.drop(columns=cols_to_drop, inplace=True)

final_data = pd.merge(merged_abc, data_c, left_on='location_name', right_on='name', how='inner')

print("\nfinal_data shape:", final_data.shape)
print("final_data columns:")
print(final_data.columns)
