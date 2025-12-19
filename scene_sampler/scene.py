from __future__ import annotations
from typing import Any, Dict, List, Optional, ClassVar, Union
from dataclasses import dataclass

from omni.isaac.core.utils.prims import create_prim, define_prim
from omni.isaac.core.utils.rotations import rot_matrix_to_quat

from geometry import DynamicObject
from pxr import Usd, UsdGeom

from pxr.Usd import Prim

import numpy as np
import os
@dataclass
class Scene():
    """Classobject for generating Isaac-Sim Scenes

    Returns:
        Scene: Scene-Object 
    """

    objects: Optional[List[DynamicObject]] = None
    table: Optional[List[DynamicObject]] = None
    names: Optional[List[str]] = None
    positions: Optional[dir[str, List]] = None
    orientations: Optional[dir[str, List]] = None
    scales: Optional[dir[str, List]] = None
    table: str = None

    work_path: str = None

    @staticmethod
    def from_dict(data_dict: Dict[str, Any],
                  root_path: str = "/World/Workstation",
                  num_objects: int = 5) -> Scene:
    
        work_path = root_path + "_0"
        _ = define_prim(work_path)

        names = []

        tables = ["Danny.usd", "Appleseed_CoffeeTable.usd", "EastRural_Table.usd", "Make.usd",
                  "Midtown.usd", "OakTableSmall.usd", "Roxana_CoffeeTable.usd", "Roxana_DiningTable.usd", "Willow.usd"]

        # table_usd = get_table_usd()

        root = "omniverse://localhost/Assets/tables_usd/"
        _table = np.random.choice(tables)

        table_usd = root + _table
        _ = create_prim(prim_path=work_path + "/" + "table", usd_path=table_usd, scale=[0.01])

        keys = np.array(list(data_dict.keys()))
        
        for _ in range(num_objects):
            try:
                random = np.random.choice(keys)
                name = data_dict[random]["name"]
                usd_path = data_dict[random]["file_path"]
                print(usd_path)
                _ = create_prim(prim_path=work_path + "/" + name,
                                    usd_path=usd_path)
                names.append(name)
            except:
                pass

        return Scene(
            names=names,
            work_path=work_path,
            table = _table
        )
    @staticmethod
    def from_dataset(data_dict: Dict[str, Any],
                     root_path: str = "/World/Workstation"):
        work_path = root_path
        _ = define_prim(work_path)

        names = []
        orientations = {}
        positions = {}
        scales = {}

        table_usd = data_dict["table"]["file_path"].rstrip(".obj") + ".usd"
        print(table_usd)
        _ = create_prim(prim_path=work_path + "/" + "table", usd_path=table_usd, scale=[0.01])
        for key in data_dict.keys():
            if key == "grasp_path":
                continue
            if key != "table":
                name = data_dict[key]["name"]
                usd_path = data_dict[key]["file_path"]
                position = data_dict[key]["position"]
                orientation = data_dict[key]["orientation"]
                scale = data_dict[key]["scale"]
                _ = create_prim(prim_path=work_path + "/" + name, usd_path=usd_path)

                positions[name] = position
                orientations[name] = orientation
                scales[name] = scale
                
                names.append(name)
        return Scene(
            names=names,
            work_path=work_path,
            positions=positions,
            orientations=orientations,
            scales=scales
        )
