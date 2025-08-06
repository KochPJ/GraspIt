from typing import Optional, Sequence

from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name

from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.materials.preview_surface import PreviewSurface
from omni.isaac.core.materials.visual_material import VisualMaterial

import omni.isaac.core.utils.bounds as bounds_utils

import numpy as np

from pxr import Usd


class DynamicObject(RigidPrim, GeometryPrim):
    """High level wrapper to create/encapsulate a dynamic object

    .. note::

        Dynamic object (Obejct shape) have collisions (Collider API) and rigid body dynamics (Rigid Body API)

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create
        name (str, optional): shortname to be used as a key by Scene class.
                                Note: needs to be unique if the object is added to the Scene.
                                Defaults to "fixed_cube".
        position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                        (with respect to its parent prim). shape is (3, ).
                                                        Defaults to None, which means left unchanged.
        orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                        (depends if translation or position is specified).
                                                        quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                        Defaults to None, which means left unchanged.
        scale (Optional[Sequence[float]], optional): local scale to be applied to the prim's dimensions. shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        color (Optional[np.ndarray], optional): color of the visual shape. Defaults to None, which means 50% gray
        size (Optional[float], optional): length of each cube edge. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): visual material to be applied to the held prim.
                                Defaults to None. If not specified, a default visual material will be added.
        physics_material (Optional[PhysicsMaterial], optional): physics material to be applied to the held prim.
                                Defaults to None. If not specified, a default physics material will be added.
        mass (Optional[float], optional): mass in kg. Defaults to None.
        density (Optional[float], optional): density. Defaults to None.
        linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
        angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.

    Example:

    .. code-block:: python

        >>> from omni.isaac.core.objects import DynamicCuboid
        >>> import numpy as np
        >>>
        >>> # create a red dynamic cube of mass 1kg at the given path
        >>> prim = DynamicCuboid(prim_path="/World/Xform/Cube", color=np.array([1.0, 0.0, 0.0]), mass=1.0)
        >>> prim
        <omni.isaac.core.objects.cuboid.DynamicCuboid object at 0x7ff14c04d990>
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "dynamic_obj",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        mass: Optional[float] = None,
        color: Optional[np.ndarray] = None,
        size: Optional[float] = None,
        use_visual_material: Optional[bool] = True,
        use_physics_material: Optional[bool] = True,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        density: Optional[float] = None,
        collision: Optional[bool] = True,
        rigid_body_physics: Optional[bool] = True,
        approximation = "convexDecomposition",
    ) -> None:
        if not is_prim_path_valid(prim_path):
           raise Exception("prim_path is not valid")
        if mass is None:
            mass = 0.02
        if size is None:
            size = 1.0
        if visible is None:
            visible = True
        if use_visual_material and visual_material is None:
            if color is None:
                color = np.array([0.5, 0.5, 0.5])
            visual_prim_path = find_unique_string_name(
                initial_name="/World/Looks/visual_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
        if use_physics_material and physics_material is None:
            static_friction = 0.2
            dynamic_friction = 1.0
            restitution = 0.0
            physics_material_path = find_unique_string_name(
                initial_name="/World/Physics_Materials/physics_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            physics_material = PhysicsMaterial(
                prim_path=physics_material_path,
                dynamic_friction=dynamic_friction,
                static_friction=static_friction,
                restitution=restitution,
            )

        # Rigid Body API
        RigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            density=density,
        )

        if rigid_body_physics:
            RigidPrim.enable_rigid_body_physics(self)
        else:
            RigidPrim.disable_rigid_body_physics(self)
        
        # Collider API
        GeometryPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            collision=collision,
        )

        if use_visual_material:
            DynamicObject.apply_visual_material(self, visual_material)
        if use_physics_material:
            DynamicObject.apply_physics_material(self, physics_material)

        GeometryPrim.set_collision_enabled(self, collision)
        
        # Collider Approximation
        if approximation is not None:
            self.set_collision_approximation(approximation)
            if approximation == "convexDecomposition":
                API = Usd.SchemaRegistry.GetAPITypeFromSchemaTypeName("PhysxConvexDecompositionCollisionAPI")
                self.prim.ApplyAPI(API)
                # self.prim.GetAttribute("physxConvexDecompositionCollision:voxelResolution").Set(500_000)
                self.prim.GetAttribute("physxConvexDecompositionCollision:errorPercentage").Set(10)
                self.prim.GetAttribute("physxConvexDecompositionCollision:hullVertexLimit").Set(32)
                self.prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Set(32)  # 2048
                self.prim.GetAttribute('physxConvexDecompositionCollision:shrinkWrap').Set(True)

        # limit the object dynamics
        API = Usd.SchemaRegistry.GetAPITypeFromSchemaTypeName("PhysxRigidBodyAPI")
        self.prim.ApplyAPI(API)
        self.prim.GetAttribute("physxRigidBody:maxLinearVelocity").Set(1.5)
        self.prim.GetAttribute("physxRigidBody:maxAngularVelocity").Set(300)
        self.prim.GetAttribute("physxRigidBody:maxContactImpulse").Set(1)
        self.prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Set(1)

        self._cache = bounds_utils.create_bbox_cache()

    def set_collision_flag(self, enabled: bool) -> None:
        self.set_collision_enabled(enabled)

    def get_collision_flag(self) -> bool:
        return self.get_collision_enabled()
    
    def compute_bb(self) -> np.array:
        return bounds_utils.compute_aabb(self._cache, prim_path=self.prim_path)
    