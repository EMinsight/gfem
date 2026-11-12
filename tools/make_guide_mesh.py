"""Build the mesh of the "Prepare a mesh for gfem" recipe with gmsh.

Usage, from the directory the mesh goes to:

    uv run --no-project --with gmsh==4.15.2 python tools/make_guide_mesh.py

It writes ``my_line.msh`` (mm): the parallel-plate line of the get-started
page, 40 mm long, 6 mm wide and 1 mm high, with a ceramic in the middle
20 mm. Volumes "feed" and "design"; surfaces "bottom" and "top" (the plates),
"port1" (x = 0) and "port2" (x = 40). The side walls are left unnamed. The
recipe shows this script block for block, and
``tests/unit/data/guide_line.msh`` is its output, made with gmsh 4.15.2.
"""

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 1)

occ = gmsh.model.occ
feed_in = occ.addBox(0, 0, 0, 10, 6, 1)  # x, y, z, then dx, dy, dz in mm
design = occ.addBox(10, 0, 0, 20, 6, 1)
feed_out = occ.addBox(30, 0, 0, 10, 6, 1)
occ.fragment([(3, feed_in)], [(3, design), (3, feed_out)])
occ.synchronize()

gmsh.model.addPhysicalGroup(3, [feed_in, feed_out], name="feed")
gmsh.model.addPhysicalGroup(3, [design], name="design")


def faces_in_plane(axis, value):
    """The surfaces that lie in the plane x, y or z (axis 0, 1, 2) = value."""
    found = []
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)  # xmin, ymin, zmin, xmax, ...
        if abs(box[axis] - value) < 1e-6 and abs(box[axis + 3] - value) < 1e-6:
            found.append(tag)
    return found


gmsh.model.addPhysicalGroup(2, faces_in_plane(2, 0), name="bottom")
gmsh.model.addPhysicalGroup(2, faces_in_plane(2, 1), name="top")
gmsh.model.addPhysicalGroup(2, faces_in_plane(0, 0), name="port1")
gmsh.model.addPhysicalGroup(2, faces_in_plane(0, 40), name="port2")

gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)  # mm
gmsh.model.mesh.generate(3)
gmsh.write("my_line.msh")
gmsh.finalize()
