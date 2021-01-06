import json
import os

TRAIN_LIST_PATH = './data/invoice/train/train_samples_list.csv'
TRAIN_BOXES_AND_TRANSCRIPTS_FOLDER = './data/invoice/train/boxes_and_transcripts'
TRAIN_IMAGES_FOLDER = './data/invoice/train/images'

VAL_BOXES_AND_TRANSCRIPTS_FOLDER = './data/invoice/val/boxes_and_transcripts'
VAL_LIST_PATH = './data/invoice/val/train_samples_list.csv'
VAL_IMAGES_FOLDER = './data/invoice/val/images'

TEST_BOXES_AND_TRANSCRIPTS_FOLDER = './data/invoice/test/boxes_and_transcripts'
TEST_LIST_PATH = './data/invoice/test/train_samples_list.csv'
TEST_IMAGES_FOLDER = './data/invoice/test/images'

FIELDS = ["company_name", "company_address", "company_tel", "company_fax", "bank_name", "branch_name", "account_name",
          "account_type", "account_number", "amount_excluding_tax", "amount_including_tax", "item_name",
          "item_quantity", "item_unit", "item_unit_amount", "item_total_amount", "tax", "company_zipcode",
          "payment_date", "issued_date", "delivery_date", "document_number", "invoice_number", "item_line_number",
          "company_department_name"]

for folder in [TRAIN_BOXES_AND_TRANSCRIPTS_FOLDER, VAL_BOXES_AND_TRANSCRIPTS_FOLDER, TEST_BOXES_AND_TRANSCRIPTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def get_label_paths(data_path):
    return [os.path.join(data_path, label_path) for label_path in os.listdir(data_path)]


def process_file(json_path, save_folder):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # parse and save to save_folder
    all_annotations = _load_annotations(data)
    filename = os.path.basename(json_path)
    with open(os.path.join(save_folder, filename.replace('.json', '.tsv')), 'w') as f:
        for i, annotation in enumerate(all_annotations):
            s = []
            s.append(str(i + 1))

            loc = sum(list(map(list, annotation["polygon"])), [])
            loc = list(map(str, loc))
            s += loc

            # TODO: replace comma with other character
            s.append(annotation['text'].replace(",", "|"))

            formal_key = annotation['formal_key']

            if formal_key in FIELDS:
                s.append(annotation['key_type'] + '_' + formal_key)
            else:
                s.append("other")

            f.write(','.join(s) + '\n')


def _load_annotations(label):
    all_annotations = []
    all_regions = label['attributes']['_via_img_metadata']['regions']

    for idx, region in enumerate(all_regions):
        region_attr = region['region_attributes']
        shape_attr = region['shape_attributes']

        try:
            if shape_attr['name'] == 'polygon':
                all_x = shape_attr['all_points_x']
                all_y = shape_attr['all_points_y']
                polygon = list(zip(all_x, all_y))
            else:
                x1 = shape_attr['x']
                y1 = shape_attr['y']
                x2 = shape_attr['width'] + x1
                y2 = shape_attr['height'] + y1
                polygon = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2)
                ]

            annotation = {
                "polygon": polygon,
                "text": str(region_attr["label"]),
                "formal_key": region_attr.get("formal_key", None),
                "key_type": region_attr.get("key_type", None)
            }

        except KeyError as e:
            continue

        all_annotations.append(annotation)
    return all_annotations


def process(data_folder, box_folder, list_path):
    label_paths = get_label_paths(os.path.join(data_folder, 'labels'))

    with open(list_path, 'w') as f:
        for i, path in enumerate(label_paths):
            process_file(path, box_folder)
            basename = os.path.basename(path)
            basename = os.path.splitext(basename)[0]
            f.write(str(i + 1) + ',' + 'invoice,' + basename + '\n')


process('data/invoice/train', TRAIN_BOXES_AND_TRANSCRIPTS_FOLDER, TRAIN_LIST_PATH)
process('data/invoice/val', VAL_BOXES_AND_TRANSCRIPTS_FOLDER, VAL_LIST_PATH)
process('data/invoice/test', TEST_BOXES_AND_TRANSCRIPTS_FOLDER, TEST_LIST_PATH)
