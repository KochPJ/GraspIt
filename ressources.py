import os
import yaml
import shutil
from shutil import move
import numpy as np

maximum = 0

def list_assets():
    objects_dict = {}
    PATH = "/share/assets"
    for index, folder in enumerate(os.listdir(PATH)):
        print(folder, index)
        _name = f"object{index}"
        objects_dict[_name] = {}
        objects_dict[_name]["file_path"] = os.path.join(PATH, folder, f"{folder}.usd")
        objects_dict[_name]["name"] = f"object{folder}"
    with open("configs/abc_objects.yaml", "w") as f:
        yaml.dump(objects_dict, f)

def convert_asset_names():
    PATH = "/share/assets"
    for index, folder in enumerate(os.listdir(PATH)):
        for item in os.listdir(os.path.join(PATH, folder)):
            print(item)
            os.system(f"mv {os.path.join(PATH, folder, item)} {os.path.join(PATH, folder)}/{folder}.usd")

def mix_textures():
    PATH = "/share/textures"
    for folder in os.listdir(PATH):
        if os.path.isdir(os.path.join(PATH, folder)):
            os.rmdir(os.path.join(PATH, folder))
    print(len(os.listdir(PATH)))


def show_depth_map():
    import matplotlib.pyplot as plt
    import numpy as np

    images = [raw for raw in os.listdir("datasets/20250128_142130") if ".npy" in raw]
    img = None
    for image in images:
        depth = np.load(os.path.join("datasets/20250128_142130", image))
        if img is None:
            img = plt.imshow(depth)
        else:
            img.set_data(depth)
        plt.pause(1)
        plt.draw()

def delete_dupes():
    for _dir in os.listdir("/mnt/logicNAS/DataSets/ModelNet40"):
        if "converted" in _dir:
            print(os.path.join(os.getcwd(), _dir))
            shutil.rmtree(os.path.join("/mnt/logicNAS/DataSets/ModelNet40", _dir))

def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    controll =  lines[0].strip()
    if controll == "OFF":
    # Read vertex and face counts
        n_vertices, n_faces, _ = map(int, lines[1].strip().split())
    
    # Read vertices
        vertices = []
        for i in range(2, 2 + n_vertices):
            vertices.append(lines[i].strip().split())
    
    # Read faces
        faces = []
        for i in range(2 + n_vertices, 2 + n_vertices + n_faces):
            face_data = list(map(int, lines[i].strip().split()))
            faces.append(face_data[1:])  # Skip the first element which is the count
    else: 
        assert "OFF" in lines[0].strip()
        n_vertices, n_faces, _ = map(int, lines[0].lstrip("OFF").split())

        vertices = []
        for i in range(1, 1 + n_vertices):
            vertices.append(lines[i].strip().split())

        faces = []
        for i in range(1 + n_vertices, 1 + n_vertices + n_faces):
            face_data = list(map(int, lines[i].strip().split()))
            faces.append(face_data[1:])
    
    return vertices, faces

def write_obj(vertices, faces, output_file):
    with open(output_file, 'w') as f:
        for v in vertices:
            f.write(f"v {' '.join(v)}\n")
        for face in faces:
            f.write(f"f {' '.join(str(idx + 1) for idx in face)}\n")  # OBJ format is 1-indexed

def convert_off_to_obj():
    PATH = "/mnt/logicNAS/DataSets/ModelNet40"
    OUTPATH = "/home/sersandr/synthData/omniverse/ModelNet40/"
    for _dir in os.listdir(PATH):
        if "converted" in _dir:
            continue
        converted = f"{_dir}_converted"
        os.makedirs(os.path.join(OUTPATH, converted), exist_ok=True)
        for folder in os.listdir(os.path.join(PATH, _dir)):
            os.makedirs(os.path.join(OUTPATH, converted, folder), exist_ok=True)
            for obj in os.listdir(os.path.join(PATH, _dir, folder)):
                input_file = os.path.join(PATH, _dir, folder, obj)
                output_file = os.path.join(OUTPATH, converted, folder, f"{obj.strip('.off')}.obj")
                print(input_file, output_file)
                vertices, faces = read_off(input_file)
                write_obj(vertices, faces, output_file)
    #vertices, faces = read_off(input_file)
    #write_obj(vertices, faces, output_file)




if __name__ == "__main__":
    path = "/home/sersandr/synthData/instant-ngp/data"
    for folder in os.listdir(path):
        if folder in {"Demo_A", "Demo_B", "Demo_C"}:
            os.makedirs(os.path.join(path, folder, "images"), exist_ok=True)
            for file in os.listdir(os.path.join(path, folder)):
                if file != "images":
                    move(os.path.join(path, folder, file), os.path.join(path, folder, "images"))
                shutil.copyfile(os.path.join(path, "parameter.json"), os.path.join(path, folder, "transforms.json"))