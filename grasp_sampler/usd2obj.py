from utlis import usd2obj
import os


def convert_usd_to_obj(directory):
    usd_to_obj_mapping = []

    # Sammle alle .usd-Dateien und prüfe, ob entsprechende .obj-Dateien existieren
    for root, dirs, files in os.walk(directory):
        usd_files = [f for f in files if f.endswith('.usd')]
        obj_files = set(f for f in files if f.endswith('.obj'))
        
        for usd_file in usd_files:
            usd_path = os.path.join(root, usd_file)
            obj_path = os.path.join(root, usd_file.replace('.usd', '.obj'))
            
            if os.path.basename(obj_path) not in obj_files:
                # Füge die zu konvertierenden Dateien zur Liste hinzu
                usd_to_obj_mapping.append((usd_path, obj_path))

    print(len(usd_to_obj_mapping))

    # Generiere die fehlenden .obj-Dateien
    for usd_path, obj_path in usd_to_obj_mapping:
        usd2obj(usd_path, obj_path)

convert_usd_to_obj('/share/assets')
