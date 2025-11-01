import sys
import traceback
import pickle
import os
import concurrent.futures
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

from font_dataset.font import load_fonts, DSFont
from font_dataset.layout import generate_font_image, TextSizeTooSmallException
from font_dataset.text import CorpusGeneratorManager, UnqualifiedFontException
from font_dataset.background import background_image_generator

# --- Arguments ---
# Usage:
#   python dataset_gen.py <shard_index> <total_shards> <train_cnt> <val_cnt> <test_cnt>
# Example:
#   python dataset_gen.py 1 8 150 20 30

if len(sys.argv) < 6:
    print("Usage: python dataset_gen.py <shard_index> <total_shards> <train_cnt> <val_cnt> <test_cnt>")
    sys.exit(1)

global_script_index = int(sys.argv[1])
global_script_index_total = int(sys.argv[2])
train_cnt = int(sys.argv[3])
val_cnt = int(sys.argv[4])
test_cnt = int(sys.argv[5])

print(f"Shard {global_script_index} / {global_script_index_total}")
print(f"Counts → train: {train_cnt}, val: {val_cnt}, test: {test_cnt}")

num_workers = 32

dataset_path = "./dataset/font_img"
os.makedirs(dataset_path, exist_ok=True)

unqualified_log_file_name = f"unqualified_font_{time.time()}.txt"
runtime_exclusion_list = []

fonts, exclusion_rule = load_fonts()
print(f"Total fonts loaded: {len(fonts)}")

# --- Split fonts into train/val/test ---
train_fonts, temp_fonts = train_test_split(fonts, test_size=0.2, random_state=42, shuffle=True)
val_fonts, test_fonts = train_test_split(temp_fonts, test_size=0.5, random_state=42, shuffle=True)

corpus_manager = CorpusGeneratorManager()
print("Corpus manager initialized.")

def add_exclusion(font: DSFont, reason: str, dataset_base_dir: str, i: int, j: int):
    print(f"Excluded font: {font.path}, reason: {reason}")
    runtime_exclusion_list.append(font.path)
    with open(unqualified_log_file_name, "a+") as f:
        f.write(f"{font.path} # {reason}\n")
    for jj in range(j + 1):
        image_file_name = f"font_{i}_img_{jj}.jpg"
        label_file_name = f"font_{i}_img_{jj}.bin"
        for name in (image_file_name, label_file_name):
            path = os.path.join(dataset_base_dir, name)
            if os.path.exists(path):
                os.remove(path)

def generate_dataset(dataset_type: str, cnt: int, fonts_subset):
    dataset_base_dir = os.path.join(dataset_path, dataset_type)
    os.makedirs(dataset_base_dir, exist_ok=True)

    def _generate_single(args):
        i, j, font = args
        print(f"Generating {dataset_type} font: {font.path} {i}/{len(fonts_subset)}, image {j}")

        if exclusion_rule(font) or font.path in runtime_exclusion_list:
            print(f"Excluded font: {font.path}")
            return

        while True:
            try:
                image_file_name = f"font_{i}_img_{j}.jpg"
                label_file_name = f"font_{i}_img_{j}.bin"

                image_file_path = os.path.join(dataset_base_dir, image_file_name)
                label_file_path = os.path.join(dataset_base_dir, label_file_name)

                if os.path.exists(image_file_path) and os.path.exists(label_file_path):
                    return

                im = next(background_image_generator())  # new background each time
                im, label = generate_font_image(im, font, corpus_manager)
                im.save(image_file_path)
                pickle.dump(label, open(label_file_path, "wb"))
                return
            except UnqualifiedFontException:
                traceback.print_exc()
                add_exclusion(font, "unqualified font", dataset_base_dir, i, j)
                return
            except TextSizeTooSmallException:
                traceback.print_exc()
                continue
            except Exception as e:
                traceback.print_exc()
                add_exclusion(font, f"other: {repr(e)}", dataset_base_dir, i, j)
                return

    work_list = []
    for i, font in enumerate(fonts_subset):
        for j in range(cnt):
            work_list.append((i, j, font))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_generate_single, work_list), total=len(work_list), desc=dataset_type))

# --- Divide fonts among shards ---
total_fonts = len(fonts)
fonts_per_shard = total_fonts // global_script_index_total
start_idx = (global_script_index - 1) * fonts_per_shard
end_idx = global_script_index * fonts_per_shard if global_script_index < global_script_index_total else total_fonts
fonts_shard = fonts[start_idx:end_idx]

# Optional: print distribution info
print(f"Using fonts {start_idx}–{end_idx} for this shard ({len(fonts_shard)} fonts).")

# --- Run generation ---
generate_dataset("train", train_cnt, train_fonts)
generate_dataset("val", val_cnt, val_fonts)
generate_dataset("test", test_cnt, test_fonts)
