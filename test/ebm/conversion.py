import pandas as pd
import math
import json
import glob
import os

# =========================================================
# CONFIG
# =========================================================

X_COL = "x"
Y_COL = "y"

# =========================================================
# FIND ALL CSV FILES
# =========================================================

csv_files = glob.glob("*.csv")

if not csv_files:
    print("No CSV files found.")
    exit()

print(f"Found {len(csv_files)} CSV files.\n")

# =========================================================
# PROCESS EACH CSV
# =========================================================

for CSV_FILE in csv_files:

    print("\n================================================")
    print(f"PROCESSING: {CSV_FILE}")
    print("================================================")

    # -----------------------------------------------------
    # OUTPUT NAMES
    # -----------------------------------------------------

    base_name = os.path.splitext(CSV_FILE)[0]

    OUTPUT_EMPTY = f"{base_name}_checkerboard_empty.json"
    OUTPUT_SOLVED = f"{base_name}_checkerboard_solved.json"
    OUTPUT_VOCAB = f"{base_name}_vocab.json"

    # -----------------------------------------------------
    # LOAD CSV
    # -----------------------------------------------------

    df = pd.read_csv(CSV_FILE,header=None,names=["x", "y"],dtype=str).fillna("")

    if X_COL not in df.columns or Y_COL not in df.columns:
        print(f"Skipping {CSV_FILE} (missing x/y columns)")
        continue

    df[X_COL] = df[X_COL].astype(str)
    df[Y_COL] = df[Y_COL].astype(str)

    # -----------------------------------------------------
    # FIND MAX LENGTHS
    # -----------------------------------------------------

    max_x_len = df[X_COL].str.len().max()
    max_y_len = df[Y_COL].str.len().max()

    print(f"Max X Length: {max_x_len}")
    print(f"Max Y Length: {max_y_len}")

    # -----------------------------------------------------
    # FIND VOCABULARY
    # -----------------------------------------------------

    all_text = "".join(df[X_COL].tolist()) + "".join(df[Y_COL].tolist())

    unique_chars = sorted(set(all_text))

    vocab = {
        "<PAD>": 0,
        "<MASK>": 1,
    }

    for i, ch in enumerate(unique_chars, start=2):
        vocab[ch] = i

    PAD = vocab["<PAD>"]
    MASK = vocab["<MASK>"]

    print(f"Unique Characters: {len(unique_chars)}")
    print(f"Vocabulary Size: {len(vocab)}")

    # -----------------------------------------------------
    # SAVE VOCAB
    # -----------------------------------------------------

    with open(OUTPUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------
    # GRID SIZE
    # -----------------------------------------------------

    required_cells = max(max_x_len, max_y_len) * 2

    grid_size = math.ceil(math.sqrt(required_cells))

    # make even for checkerboard symmetry
    if grid_size % 2 != 0:
        grid_size += 1

    total_cells = grid_size * grid_size

    print(f"Grid Size: {grid_size} x {grid_size}")
    print(f"Total Cells: {total_cells}")

    # -----------------------------------------------------
    # ENCODE FUNCTION
    # -----------------------------------------------------

    def encode(text):
        return [vocab[ch] for ch in text]

    # -----------------------------------------------------
    # CREATE DATASETS
    # -----------------------------------------------------

    empty_dataset = []
    solved_dataset = []

    for idx, row in df.iterrows():

        x = row[X_COL]
        y = row[Y_COL]

        x_tokens = encode(x)
        y_tokens = encode(y)

        solved_grid = [
            [PAD for _ in range(grid_size)]
            for _ in range(grid_size)
        ]

        empty_grid = [
            [PAD for _ in range(grid_size)]
            for _ in range(grid_size)
        ]

        x_ptr = 0
        y_ptr = 0

        # -------------------------------------------------
        # CHECKERBOARD
        # -------------------------------------------------

        for r in range(grid_size):
            for c in range(grid_size):

                # EVEN = X
                if (r + c) % 2 == 0:

                    if x_ptr < len(x_tokens):

                        token = x_tokens[x_ptr]

                        solved_grid[r][c] = token
                        empty_grid[r][c] = token

                        x_ptr += 1

                # ODD = Y
                else:

                    if y_ptr < len(y_tokens):

                        token = y_tokens[y_ptr]

                        solved_grid[r][c] = token

                        # hide y
                        empty_grid[r][c] = MASK

                        y_ptr += 1

        # -------------------------------------------------
        # SAVE SAMPLE
        # -------------------------------------------------

        empty_dataset.append({
            "id": int(idx),
            "x": x,
            "y": y,
            "grid": empty_grid
        })

        solved_dataset.append({
            "id": int(idx),
            "x": x,
            "y": y,
            "grid": solved_grid
        })

    # -----------------------------------------------------
    # SAVE JSON FILES
    # -----------------------------------------------------

    with open(OUTPUT_EMPTY, "w", encoding="utf-8") as f:
        json.dump(empty_dataset, f, indent=2)

    with open(OUTPUT_SOLVED, "w", encoding="utf-8") as f:
        json.dump(solved_dataset, f, indent=2)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print(f"\nSaved:")
    print(f" - {OUTPUT_EMPTY}")
    print(f" - {OUTPUT_SOLVED}")
    print(f" - {OUTPUT_VOCAB}")

    print(f"Samples Processed: {len(df)}")

print("\n========================================")
print("ALL CSV FILES PROCESSED")
print("========================================")
