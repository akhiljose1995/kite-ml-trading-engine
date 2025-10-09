import pandas as pd
# Show all columns, all rows, and full column content
pd.set_option('display.max_columns', None)       # Show all columns
pd.set_option('display.max_rows', None)          # Show all rows (optional)
pd.set_option('display.max_colwidth', None)      # Don't truncate column contents
pd.set_option('display.width', 0)                # Auto-detect console width

# Replace with your actual file path
df = pd.read_csv('STATE_BANK_OF_INDIA.csv')

# View the first 5 rows
print(df.head())

for col in df.columns:
    if col == "date":
        continue
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        print(f"{col}: {df[col].dropna().unique().tolist()}")

num_cols = df.select_dtypes(include='number').columns
print(df[num_cols].describe())