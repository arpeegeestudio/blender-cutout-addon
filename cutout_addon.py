bl_info = {
    "name": "Cutout Filter",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import subprocess
import sys
import importlib.util

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")

def check_and_install_packages():
    packages = ['opencv-python', 'scikit-learn']
    for package in packages:
        if importlib.util.find_spec(package.replace("-", "_")) is None:
            install_package(package)

check_and_install_packages()

import cv2
import numpy as np
from sklearn.cluster import KMeans
import tempfile

def cluster(im, n_clusters):
    im = im.reshape((im.shape[0] * im.shape[1], 3))
    km = KMeans(n_clusters=n_clusters, random_state=0)
    km.fit(im)
    counts = {}
    reps = km.cluster_centers_
    for i in range(len(im)):
        if km.labels_[i] not in counts:
            counts[km.labels_[i]] = {}
        rgb = tuple(im[i])
        if rgb not in counts[km.labels_[i]]:
            counts[km.labels_[i]][rgb] = 0
        counts[km.labels_[i]][rgb] += 1
    for label, hist in counts.items():
        flat = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        reps[label] = flat[0][0]
    return km.cluster_centers_, km.labels_

def remap_colors(im, reps, labels):
    orig_shape = im.shape
    im = im.reshape((im.shape[0] * im.shape[1], 3))
    for i in range(len(im)):
        im[i] = reps[labels[i]]
    return im.reshape(orig_shape)

def find_contours(im, reps, min_area):
    contours = []
    for rep in reps:
        mask = cv2.inRange(im, rep-1, rep+1)
        conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cont in conts:
            area = cv2.contourArea(cont)
            if area >= min_area:
                contours.append((area, cont, rep))
    contours.sort(key=lambda x: x[0], reverse=True)
    return contours

def apply_cutout_filter(image_path, n_clusters=5, min_area=50, poly_epsilon=3, final_blur=False):
    orig = cv2.imread(image_path)
    im = orig.copy()
    im = cv2.GaussianBlur(im, (3, 3), 0)
    reps, labels = cluster(im, n_clusters)
    im = remap_colors(im, reps, labels)
    contours = find_contours(im, reps, min_area)
    canvas = np.zeros(orig.shape, np.uint8)
    for area, cont, rep in contours:
        approx = cv2.approxPolyDP(cont, poly_epsilon, True)
        cv2.drawContours(canvas, [approx], -1, rep, -1)
    if final_blur:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename, canvas)
    return temp_filename

class CutoutOperator(bpy.types.Operator):
    bl_idname = "object.cutout_operator"
    bl_label = "Apply Cutout Filter"

    def execute(self, context):
        obj = context.object
        if obj.type == 'MESH':
            material = obj.active_material
            if material and material.use_nodes:
                nodes = material.node_tree.nodes
                for node in nodes:
                    if node.type == 'TEX_IMAGE':
                        image = node.image
                        if image:
                            image_path = bpy.path.abspath(image.filepath)
                            output_path = apply_cutout_filter(image_path)
                            image.filepath = output_path
                            image.reload()
                            self.report({'INFO'}, "Cutout filter applied successfully!")
                            return {'FINISHED'}
                self.report({'ERROR'}, "No image texture found in the material.")
            else:
                self.report({'ERROR'}, "Selected object has no material or it doesn't use nodes.")
        else:
            self.report({'ERROR'}, "Selected object is not a mesh.")
        return {'CANCELLED'}

def menu_func(self, context):
    self.layout.operator(CutoutOperator.bl_idname)

def register():
    bpy.utils.register_class(CutoutOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(CutoutOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()
