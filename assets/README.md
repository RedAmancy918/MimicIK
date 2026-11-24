# Kinematic models

`airbot.urdf` preserves the joint transforms, axes, names and limits from the
original AIRBOT file, with visual/collision meshes and inertials removed. It is
for FK only, not collision checking or dynamic simulation. Joint order is
reported by the FK API; the TCP is `end_link`.

Original URDF SHA256: `f06fa457b79c5b87bb80aa4e111fd5a9433ff7fb2a12ef2f28c79a7fa4065b1c`.

`toy6.urdf` is an independently generated planar six-joint test chain matching
`examples/make_demo_data.py`. It is not an AIRBOT model.
