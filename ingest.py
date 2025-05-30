import os


def create_metadata_folder_from_json(data, output_metadata_dir="metadata_folder"):
    os.makedirs(output_metadata_dir, exist_ok=True)

    def convert_concadia_entry_to_metadata(entry):
        filename = entry.get("filename", "unknown")
        picture_id = f"#/pictures/{os.path.splitext(filename)[0]}"
        metadata = {
            "section_header": None,
            "above_text": None,
            "caption": entry.get("caption", {}).get("raw", None),
            "picture_id": picture_id,
            "footnote": None,
            "below_text": entry.get("context", {}).get("raw", None),
        }
        return metadata

    for entry in data.get("images", []):
        metadata = convert_concadia_entry_to_metadata(entry)
        filename = entry.get("filename", "unknown")
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_metadata_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            for key, value in metadata.items():
                val_str = value if value is not None else "null"
                f.write(f"{key}: {val_str}\n")
    print(f"Metadata files created in '{output_metadata_dir}' directory.")


# Usage example:
import json

with open(
    "C:\\Users\\pssan\\Desktop\\sound\\wiki_split.json", "r", encoding="utf-8"
) as f:
    concadia_data = json.load(f)
create_metadata_folder_from_json(concadia_data)
