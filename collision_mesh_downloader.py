# Zealan's collision mesh downloader that converts RLUtilities meshes to the RocketSim .cmf format
# Written 2024/6/26

from urllib.request import urlretrieve
import os
import numpy as np
import copy
import struct

class Mesh:
	def __init__(self, verts, tris):
		self.verts = verts
		self.tris = tris
		
	def translate(self, offset):
		result = copy.deepcopy(self)
		result.verts += offset
		return result
		
	def scale(self, scale):
		result = copy.deepcopy(self)
		result.verts *= scale
		return result
	
	def flip_normals(self):
		result = copy.deepcopy(self)
		for i in range(len(result.tris)):
			tri = result.tris[i]
			new_tri = [ tri[2], tri[1], tri[0] ]
			result.tris[i] = new_tri
		return result
		
	def write_to_cmf(self):
		result = b""
		result += struct.pack("I", len(self.tris))
		result += struct.pack("I", len(self.verts))
		
		for tri in self.tris:
			for i in range(3):
				result += struct.pack("i", tri[i])
		for vert in self.verts:
			for i in range(3):
				result += struct.pack("f", vert[i])
				
		return result

def main():
	mesh_url_base = "https://github.com/samuelpmish/RLUtilities/raw/develop/assets/soccar/"
	mesh_name_prefix = "soccar_"
	mesh_names = ["corner", "goal", "ramps_0", "ramps_1"]
	verts_suffix = "_vertices"
	ids_suffix = "_ids"
	extension = ".bin"
	
	output_dir = "./rlut_meshes/"
	os.makedirs(output_dir, exist_ok=True)
	
	for mesh_name in mesh_names:
		for i in range(2):
			is_verts = (i == 0)
			filename = mesh_name_prefix + mesh_name + (verts_suffix if is_verts else ids_suffix) + extension
			
			if os.path.exists(output_dir + filename):
				print("Skipping \"{}\" (already exists)".format(filename))
				continue
			
			url = mesh_url_base + filename
			print("Downloading \"{}\"...".format(url))
			
			urlretrieve(url, output_dir + filename)
			
	base_meshes = {}
	for mesh_name in mesh_names:
		verts_path = output_dir + mesh_name_prefix + mesh_name + verts_suffix + extension
		ids_path = output_dir + mesh_name_prefix + mesh_name + ids_suffix + extension
		
		verts = np.fromfile(open(verts_path, "rb"), dtype=np.float32)
		ids = np.fromfile(open(ids_path, "rb"), dtype=np.uint32)
		
		if len(verts) % 3 != 0:
			raise Exception("Wrong verts count for mesh \"{}\": {}".format(verts_path, len(verts)))
		if len(ids) % 3 != 0:
			raise Exception("Wrong ids count for mesh \"{}\": {}".format(ids_path, len(ids)))
		
		base_meshes[mesh_name] = Mesh(verts.reshape((-1, 3)), ids.reshape((-1, 3)))
	
	# https://github.com/samuelpmish/RLUtilities/blob/67d62e1bad72b1a8ad33c6d7b62f80cf0c1f6f24/src/simulation/field.cc#L64
	flip_x =  [-1,  1,  1]
	flip_y =  [ 1, -1,  1]
	flip_xy = [-1, -1,  1]
	meshes = [
		base_meshes["corner"],
		base_meshes["corner"].scale(flip_x).flip_normals(),
		base_meshes["corner"].scale(flip_y).flip_normals(),
		base_meshes["corner"].scale(flip_xy),
		
		base_meshes["goal"].translate([0, -5120, 0]),
		base_meshes["goal"].translate([0, -5120, 0]).scale(flip_y).flip_normals(),
		
		base_meshes["ramps_0"],
		base_meshes["ramps_0"].scale(flip_x).flip_normals(),
		base_meshes["ramps_1"],
		base_meshes["ramps_1"].scale(flip_x).flip_normals()
	]
	
	# Scale to BulletPhysics units
	for i in range(len(meshes)):
		meshes[i] = meshes[i].scale(1/50)
	
	output_dir = "./collision_meshes/soccar/"
	os.makedirs(output_dir, exist_ok=True)
	
	for i in range(len(meshes)):
		path = output_dir + "mesh_{}.cmf".format(i)
		print("Saving mesh to {}...".format(path))
		with open(path, "wb") as f:
			f.write(meshes[i].write_to_cmf())
	print("Done!")
	

if __name__ == "__main__":
	main()