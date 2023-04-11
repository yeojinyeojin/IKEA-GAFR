import json

def find_key_name(keyset, name):
    for k in keyset:
        if k.startswith(name):
            return k
    raise ValueError(f"key not found {name}")


if __name__ == '__main__':

    with open("../annotator/ikea_od_two.json", 'r') as f:
        annot_1 = json.load(f)

    with open("../annotator/ikea_od_two_3.json", 'r') as f:
        annot_2 = json.load(f)

    annot_2_start = 500
    annot_2_end = 600

    keyset = annot_2['_via_img_metadata'].keys()
    for ind in range(annot_2_start, annot_2_end + 1):
        img_name = find_key_name(keyset=keyset, name=f"{ind:05d}.png")
        annot_1['_via_img_metadata'][img_name] = annot_2['_via_img_metadata'][img_name]

    with open("../annotator/ikea_od_two_merged.json", "w") as f:
        json.dump(annot_1, f, indent=6)
