"""Extract annotations from annotations_file_path and outputs it in out dir in yolo format"""

import os.path
from xml.dom import minidom

out_dir = "./out"
annotations_file_path = "./annotations.xml"

# Constants
label_dict = {"cross": 0, "circle": 1, "triangle": 2, "square": 3}
bbox_type = tuple[int, int, int, int]


def main():
    # create out dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file = minidom.parse(annotations_file_path)
    images = file.getElementsByTagName("image")

    for image in images:
        width = int(image.getAttribute("width"))
        height = int(image.getAttribute("height"))
        name = os.path.splitext(image.getAttribute("name"))[0]  # first ele is file name
        bboxes = image.getElementsByTagName("box")

        with open(os.path.join(out_dir, name + ".txt"), "w") as label_file:
            # iterate over pairs of bbox and keypoints for every sperm
            for bbox in bboxes:
                class_index = label_dict[bbox.getAttribute("label")]

                xtl, ytl, xbr, ybr = prepare_bbox(bbox)

                w = xbr - xtl
                h = ybr - ytl
                x_cen_norm = (xtl + (w / 2)) / width
                y_cen_norm = (ytl + (h / 2)) / height
                w_norm = w / width
                h_norm = h / height

                dataset_label = f"{class_index} {x_cen_norm} {y_cen_norm} {w_norm} {h_norm} "  # last space in str is necessary

                dataset_label = dataset_label.rstrip() + "\n"
                label_file.write(dataset_label)

    print("Finished writing all label files.")


def prepare_bbox(bbox: minidom.Element) -> bbox_type:
    xtl = int(float(bbox.getAttribute("xtl")))
    ytl = int(float(bbox.getAttribute("ytl")))
    xbr = int(float(bbox.getAttribute("xbr")))
    ybr = int(float(bbox.getAttribute("ybr")))
    return xtl, ytl, xbr, ybr


if __name__ == "__main__":
    main()
