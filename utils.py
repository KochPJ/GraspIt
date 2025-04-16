from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Usd, UsdGeom, Gf, Sdf, UsdLux, UsdShade
import omni.usd
from contextlib import contextmanager

def get_current_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()

def get_table_usd() -> str:
    nucleus_server = get_assets_root_path()
    asset_folder = nucleus_server + "/Isaac/Samples/Examples/FrankaNutBolt/"
    table_usd = asset_folder + "SubUSDs/Shop_Table/Shop_Table.usd"
    return table_usd


def new_omniverse_stage() -> Usd.Stage:
    """Creates a new Omniverse USD stage.
    
    This method creates a new Omniverse USD stage.  This will clear the active
    omniverse stage, replacing it with a new one.

    Returns:
        Usd.Stage:  The Omniverse USD stage.
    """

    try:
        import omni.usd
    except ImportError:
        raise ImportError("Omniverse not found.  This method is unavailable.")

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    return stage


def add_dome_light(stage: Usd.Stage, path: str, intensity: float = 1000, 
        angle: float = 180, exposure: float=0.) -> UsdLux.DomeLight:
    """Adds a dome light to a USD stage.
    
    Args:
        stage (Usd.Stage): The USD stage to modify.
        path (str): The path to add the USD prim.
        intensity (float): The intensity of the dome light (default 1000).
        angle (float): The angle of the dome light (default 180)
        exposure (float): THe exposure of the dome light (default 0)

    Returns:
        UsdLux.DomeLight:  The created Dome light.
    """

    light = UsdLux.DomeLight.Define(stage, path)

    # intensity
    light.CreateIntensityAttr().Set(intensity)
    light.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    light.CreateExposureAttr().Set(exposure)

    # cone angle
    shaping = UsdLux.ShapingAPI(light)
    shaping.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr().Set(angle)
    shaping.CreateShapingConeSoftnessAttr()
    shaping.CreateShapingFocusAttr()
    shaping.CreateShapingFocusTintAttr()
    shaping.CreateShapingIesFileAttr()

    return light

def add_sphere_light(stage: Usd.Stage, path: str, intensity=30000, 
        radius=50, angle=180, exposure=0.):
    """Adds a sphere light to a USD stage.
    
    Args:
        stage (Usd.Stage): The USD stage to modify.
        path (str): The path to add the USD prim.
        radius (float): The radius of the sphere light
        intensity (float): The intensity of the sphere light (default 1000).
        angle (float): The angle of the sphere light (default 180)
        exposure (float): THe exposure of the sphere light (default 0)

    Returns:
        UsdLux.SphereLight:  The created sphere light.
    """

    light = UsdLux.SphereLight.Define(stage, path)

    # intensity
    light.CreateIntensityAttr().Set(intensity)
    light.CreateRadiusAttr().Set(radius)
    light.CreateExposureAttr().Set(exposure)

    # cone angle
    shaping = UsdLux.ShapingAPI(light)
    shaping.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr().Set(angle)
    shaping.CreateShapingConeSoftnessAttr()
    shaping.CreateShapingFocusAttr()
    shaping.CreateShapingFocusTintAttr()
    shaping.CreateShapingIesFileAttr()

    return light