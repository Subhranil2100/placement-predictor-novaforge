import pandas as pd
import os

print("=" * 80)
print("📋 EXCEL STRUCTURE DIAGNOSTIC")
print("=" * 80)

# Find Excel file
possible_paths = [
    "Placement_Database.xlsx",
    "../docs/Placement_Database.xlsx",
    r"C:\Users\ASUS\Desktop\Websies\placement-prediction\backend\Placement_Database.xlsx"
]

excel_path = None
for path in possible_paths:
    if os.path.exists(path):
        excel_path = path
        print(f"\n✅ Found at: {os.path.abspath(path)}")
        break

if excel_path is None:
    print("\n❌ Excel not found!")
    exit(1)

# Read without header to see actual structure
print("\n" + "=" * 80)
print("RAW DATA (First 5 rows - no header processing):")
print("=" * 80)

df_raw = pd.read_excel(excel_path, header=None)
print(f"\nShape: {df_raw.shape}")
print("\nFirst 5 rows:\n")
print(df_raw.head(5).to_string())

# Show column indices
print("\n" + "=" * 80)
print("COLUMN INDICES:")
print("=" * 80)
print(f"\nTotal columns: {len(df_raw.columns)}")
print(f"Columns: {list(df_raw.columns)}")

# Try with header=0 (first row as header)
print("\n" + "=" * 80)
print("WITH HEADER ROW 0:")
print("=" * 80)

df_h0 = pd.read_excel(excel_path, header=0)
print(f"\nColumn names:\n")
for i, col in enumerate(df_h0.columns):
    print(f"{i}: '{col}'")

# Try with header=[0,1] (first two rows as header)
print("\n" + "=" * 80)
print("WITH HEADER ROWS [0,1]:")
print("=" * 80)

try:
    df_h01 = pd.read_excel(excel_path, header=[0, 1])
    print(f"\nMulti-level columns:")
    for i, col in enumerate(df_h01.columns):
        print(f"{i}: {col}")
except Exception as e:
    print(f"Error: {e}")

# Try skipping first row
print("\n" + "=" * 80)
print("SKIPPING FIRST ROW (skiprows=1):")
print("=" * 80)

df_skip1 = pd.read_excel(excel_path, skiprows=1)
print(f"\nColumn names:\n")
for i, col in enumerate(df_skip1.columns):
    print(f"{i}: '{col}'")

print(f"\nFirst 3 rows of data:")
print(df_skip1.head(3).to_string())

print("\n" + "=" * 80)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 80)