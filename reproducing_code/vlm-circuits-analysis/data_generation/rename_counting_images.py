import os
import glob
import sys
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

print(f"Repository root: {REPO_ROOT}")

# def get_counts_and_objects(image_name: str):
#     """
#     Given a counting-task image name such as
#     '3_apple_2_boy_1_coin.png' or
#     'An image of 3 apple, 2 boy, 1 coin._seed_42.png',
#     return a list of tuples (count, object).
#     """
#     image_name = os.path.basename(image_name)
#     return re.findall(r"(\d+)\s*_?\s*([a-z]+)", image_name.lower())
# from object_counting_utils import get_counts_and_objects

def get_counts_and_objects(image_name):
    """
    Given either:
      - a counting-task image name such as "3_apple_2_boy_1_coin.png"
      - or a raw generated filename such as
        "An image of 3 apple, 2 boy, 1 coin._seed_42.png"

    return a list of tuples of the form (count, object) for each object in the image.
    """
    import os
    import re

    image_name = os.path.basename(image_name)

    # First try the final normalized format: 3_apple_2_boy_1_coin.png
    matches = re.findall(r"(\d+)_([a-z]+)(?:_|\.png$)", image_name)
    if matches:
        return matches

    # Otherwise try the raw prompt-based format:
    # "An image of 3 apple, 2 boy, 1 coin._seed_42.png"
    image_name = re.sub(r"_seed_\d+\.png$", "", image_name, flags=re.IGNORECASE)
    image_name = re.sub(r"\.png$", "", image_name, flags=re.IGNORECASE)
    image_name = image_name.lower()

    matches = re.findall(r"(\d+)\s+([a-z]+)", image_name)
    return matches

# INPUT_DIR = os.path.join($HOME, "data", "counting", "raw_images", "1")
INPUT_DIR = os.path.join(os.environ["HOME"], "data", "counting", "raw_images", "1")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "counting", "images")

print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_list = glob.glob(os.path.join(INPUT_DIR, "*.png"))

print(f"Found {len(image_list)} images")

for idx, image_path in enumerate(image_list):
    counts_and_objects = get_counts_and_objects(image_path)

    # Build correct filename: 3_apple_2_boy_1_coin.png
    new_name = "_".join(
        f"{count}_{obj}" for count, obj in counts_and_objects if int(count) != 0
    ) + ".png"

    new_path = os.path.join(OUTPUT_DIR, new_name)

    # Avoid overwriting
    if os.path.exists(new_path):
        new_path = os.path.join(
            OUTPUT_DIR, new_name.replace(".png", f"_x{idx}.png")
        )

    os.rename(image_path, new_path)

print("Done.")