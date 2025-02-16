import os
import json

def generate_json(folders, output_file, num_images_per_folder):
    json_data_list = []
    question_id = 0

    for folder in folders:
        folder_images = os.listdir(folder)
        num_images = min(num_images_per_folder, len(folder_images))

        for i in range(num_images):
            image_path = os.path.join(folder, folder_images[i])
            ps_text = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying special attention to the edges of the objects, capturing some unnatural edges and perspective relationships, some incorrect semantics, unnatural lighting and darkness etc."
            json_data = {
                "image": image_path,
                "text": ps_text
            }
            json_data_list.append(json_data)
            question_id += 1

    with open(output_file, 'w') as json_file:
        for item in json_data_list:
            json.dump(item, json_file)
            json_file.write('\n')

folders = [
    "/data03/xzp/dataset/aigc_inpaint/SDXL/image",
]
output_file = "./playground/eval_questions.jsonl"
num_images_per_folder = 99999 
generate_json(folders, output_file, num_images_per_folder)


