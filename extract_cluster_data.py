import json
import shutil
import os

case_pmc_map = {
    "2925363e-d3b4-5986-a5b5-82c2d0cb433c": "PMC4782470",
    "29f2b128-9663-58b7-8182-5ea2d3232a32": "PMC11795237",
    "32539ed7-90c4-5e93-81d9-03a7a32e4f05": "PMC10762869",
    "3770ef6e-f9d3-5eba-a388-55f2e644e0d1": "PMC4784185",
    "59ba3305-ec05-5873-9a45-4a766c6609fc": "PMC10077807",
    "6b135fab-db47-50e4-922f-44ef488f9344": "PMC10392937",
    "8b9ffaa0-4922-5612-bebc-c0a3b37ef649": "PMC6360499",
    "b2285440-a1f0-5f28-987e-fcfd6317c2dd": "PMC5101502",
    "c41f73cd-46b1-51fd-b59f-44445854b671": "PMC7428098",
    "cca3a3bf-c1c3-5fe1-9597-f0854693c087": "PMC10772306"
}

data_dir = "subset_dataset/data"
image_src_dir = "subset_dataset/images"
dest_dir = "cluster_19_scimitar_syndrome"
image_dest_dir = os.path.join(dest_dir, "images")

metadata_output = []

for case_id, pmc_id in case_pmc_map.items():
    json_path = os.path.join(data_dir, f"{pmc_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            abstract = data.get('abstract', 'No abstract found.')
            images = []
            for case_data in data.get('cases', []):
                image_filename = case_data.get('file')
                if image_filename:
                    src_image_path = os.path.join(image_src_dir, image_filename)
                    if os.path.exists(src_image_path):
                        shutil.copy(src_image_path, image_dest_dir)
                        images.append(image_filename)
            
            metadata_output.append({
                "case_id": case_id,
                "pmc_id": pmc_id,
                "abstract": abstract,
                "images": images
            })

with open(os.path.join(dest_dir, "cluster_19_metadata.json"), 'w') as f:
    json.dump(metadata_output, f, indent=4)

print(f"Extracted data for {len(metadata_output)} cases.")
