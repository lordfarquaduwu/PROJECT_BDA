import pandas as pd

SRC = r"C:\BDA_Project\online+retail+ii\online_retail_II.xlsx"
OUT = r"C:\BDA_Project\online_retail_II.csv"

# StockCode has values like "85123A", so read the ID columns as text
sheets = pd.read_excel(SRC, sheet_name=None,
                       dtype={"Invoice": str, "StockCode": str})

for name, sheet in sheets.items():
    print(f"sheet '{name}': {len(sheet):,} rows")

df = pd.concat(sheets.values(), ignore_index=True)
df["Customer ID"] = df["Customer ID"].astype("Int64")  # 13085.0 -> 13085

print(f"total: {len(df):,} rows")
print(df.dtypes)
df.to_csv(OUT, index=False)
print("written:", OUT)