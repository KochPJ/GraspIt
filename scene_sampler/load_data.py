import os
import shutil
import glob

class Data:
    def __init__(self, path):
        self.path = path
        self.count = 0
        self.max_count = len(os.listdir(self.path)) - 1
    
    def __next__(self):
        if self.count > self.max_count:
            raise StopIteration
        ret = DataIterator(os.path.join(self.path, f"scene_{self.count}"))
        self.count += 1
        return ret
    
    def __iter__(self):
        if self.count > self.max_count:
            raise StopIteration
        ret = DataIterator(os.path.join(self.path, f"scene_{self.count}"))
        self.count += 1
        return ret

    

class DataIterator:
    def __init__(self, path):
        self.path = path
        self.count = 0
        self.max_count = len(os.listdir(self.path)) - 2
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.count > self.max_count:
            raise StopIteration
        ret = {}
        path = os.path.join(self.path, f"frame_{self.count}")
        ret["camera_params"] = os.path.join(path, glob.glob(os.path.join(path, "camera_params_*.json"))[0])
        ret["depth"] = os.path.join(path, "depth.png")
        ret["rgb"] = os.path.join(path, glob.glob(os.path.join(path, "rgb_*.png"))[0])
        ret["segmentation"] = os.path.join(path, glob.glob(os.path.join(path, "semantic_segmentation_*.png"))[0])
        ret["segmentation_labels"] = os.path.join(path, glob.glob(os.path.join(path, "semantic_segmentation_labels_*.json"))[0])
        self.count += 1
        return ret



if __name__ == "__main__":
    dataset = Data("/dataset")
    while True:
        try:
            scene = next(dataset)
            for data in scene:
                print(data)
        except StopIteration:
            break
    
    print("###################################################")
