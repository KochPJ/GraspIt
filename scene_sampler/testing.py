import open3d as o3d
import os
import shutil
import glob

def main():
    root = "/share/assets"
    mesh_root = glob.glob(os.path.join(root, "*"))
    for mesh in mesh_root:
        mesh_path = os.path.join(mesh, f"{os.path.basename(mesh)}.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals

        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
