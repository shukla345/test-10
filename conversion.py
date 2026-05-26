import csv
import os

def hex_to_binary(hex_str):
    hex_str = hex_str.strip().lstrip('0x').lstrip('0X')
    value = int(hex_str, 16)
    bit_length = len(hex_str) * 4
    return format(value, f'0{bit_length}b')

def text_to_binary(text):
    return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))

def convert_csv(input_file, output_file):
    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    converted_rows = []

    for i, row in enumerate(rows):
        if len(row) < 2:
            print(f"Skipping row {i+1}: not enough columns -> {row}")
            continue

        hex_col = row[0].strip()
        text_col = row[1]

        try:
            binary_hex = hex_to_binary(hex_col)
        except ValueError:
            print(f"Skipping row {i+1}: invalid hex value '{hex_col}'")
            continue

        binary_text = text_to_binary(text_col)
        converted_rows.append([binary_hex, binary_text])

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(converted_rows)

    print(f"Done! Converted {len(converted_rows)} rows -> '{output_file}'")


# --- Run for all CSVs in current folder ---
csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and not f.endswith('_train.csv')]

for file in csv_files:
    base_name = os.path.splitext(file)[0]   # removes .csv
    output_name = f"{base_name}_train.csv"
    convert_csv(file, output_name)
