from __future__ import annotations
from typing import Any, Dict, List, Optional, ClassVar, Union
from dataclasses import dataclass

from omni.isaac.core.utils.prims import create_prim, define_prim
from omni.isaac.core.utils.rotations import rot_matrix_to_quat

from geometry import DynamicObject
from utils import get_table_usd, get_current_stage
from pxr import Usd, UsdGeom

from pxr.Usd import Prim

import numpy as np
import os
@dataclass
class Scene():

    objects: Optional[List[DynamicObject]] = None
    table: Optional[List[DynamicObject]] = None
    names: Optional[List[str]] = None

    work_path: str = None


    @staticmethod
    def custom_scene(root_path:str = "/World/Workstation",
                     custom_obj: str = "") -> Scene:
        work_path = root_path + "_0"
        _ = define_prim(work_path)

        name = custom_obj.split("/")[-1].split(".")[0]

        table_usd = get_table_usd()
        _ = create_prim(prim_path=work_path + "/" + "table", usd_path=table_usd, scale=[0.01])
        _ = create_prim(prim_path=work_path + "/" + name, usd_path=custom_obj)
        names = [name]

        return Scene(
            names=names,
            work_path=work_path
        )



    @staticmethod
    def from_dict(data_dict: Dict[str, Any],
                  root_path: str = "/World/Workstation",
                  num_objects: int = 5) -> Scene:
    
        work_path = root_path + "_0"
        _ = define_prim(work_path)

        names = []

        table_usd = get_table_usd()
        _ = create_prim(prim_path=work_path + "/" + "table", usd_path=table_usd, scale=[0.01])


        for count in range(num_objects):
            try:
                random = np.random.randint(0, 9999)
                name = data_dict[f"object{random}"]["name"]
                usd_path = data_dict[f"object{random}"]["file_path"]
                print(usd_path)
                _ = create_prim(prim_path=work_path + "/" + name,
                                    usd_path=usd_path)
                names.append(name)
            except:
                pass

        return Scene(
            names=names,
            work_path=work_path
        )
