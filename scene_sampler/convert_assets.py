# Save this file as batch_convert.py
# Run it headlessly via terminal: ./python.sh batch_convert.py

import os
import asyncio
import omni
from omni.isaac.kit import SimulationApp

def progress_callback(current_step: int, total: int):
    pass

async def convert_single_file(converter_manager, context, input_path, output_path):
    import omni.kit.asset_converter
    """Handles the async conversion of a single asset."""
    task = converter_manager.create_converter_task(
        input_path, 
        output_path, 
        progress_callback, 
        context
    )
    success = await task.wait_until_finished()
    return success, task.get_status()

async def batch_process_directories(root_dir):
    # 3. Get the 2023 asset converter framework instances
    converter_manager = omni.kit.asset_converter.get_instance()
    context = omni.kit.asset_converter.AssetConverterContext()
    
    # Configure context settings for simulation stability
    context.merge_all_meshes = True        
    context.keep_all_materials = True      
    context.ignore_animations = True       
    context.ignore_camera = True
    context.ignore_light = True

    print(f"Scanning directory: {root_dir} for .obj meshes...")
    
    # 4. Use os.walk to recursively crawl all nested folders
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.obj'):
                input_file_path = os.path.join(dirpath, filename)
                
                # Create output filename swapping .obj for .usd in the same nested folder
                output_file_path = os.path.splitext(input_file_path)[0] + ".usd"
                
                print(f"\n[STARTING] {filename}")
                print(f" -> Input:  {input_file_path}")
                print(f" -> Output: {output_file_path}")
                
                try:
                    # Run the conversion task sequentially to prevent memory/GPU crashes
                    success, status = await convert_single_file(
                        converter_manager, context, input_file_path, output_file_path
                    )
                    
                    if success:
                        print(f"[SUCCESS] Converted {filename}")
                    else:
                        print(f"[FAILED] Could not convert {filename}. Status: {status}")
                        
                except Exception as e:
                    print(f"[ERROR] Exception occurred while processing {filename}: {e}")

if __name__ == "__main__":
    simulation_app = SimulationApp({"headless": True})
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")

    # Change this to the absolute or relative path of your GraspNet dataset directory
    target_root_directory = "/share/realassets/"
    
    # Execute the asynchronous batch processing loop
    asyncio.get_event_loop().run_until_complete(batch_process_directories(target_root_directory))
    
    print("\nBatch processing complete!")
    
    # Clean up and close down Isaac Sim cleanly
    simulation_app.close()