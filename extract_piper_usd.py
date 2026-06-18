from pxr import Usd

USD_PATH = "/home/cscvlab/lht/ATEC2026_Simulation_Challenge/atec_robot_model/robot/piper/piper.usd"

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise RuntimeError(f"Failed to open USD: {USD_PATH}")

# 尽量加载所有引用和 payload
stage.Load()

print("USD:", USD_PATH)

JOINT_NAMES = {
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7", "joint8",
}

EE_NAMES = {
    "gripper_base",
    "link6",
    "joint6",
    "joint7",
    "joint8",
}

print("\n==================== ALL JOINT-LIKE PRIMS ====================")

for prim in stage.TraverseAll():
    name = prim.GetName()
    path = str(prim.GetPath())
    type_name = prim.GetTypeName()

    lower = (name + " " + path + " " + type_name).lower()

    is_joint_like = (
        "joint" in lower
        or "revolute" in lower
        or "prismatic" in lower
        or "fixedjoint" in lower
        or "physicsjoint" in lower
    )

    if not is_joint_like:
        continue

    print("\n" + "=" * 100)
    print("prim path:", path)
    print("name     :", name)
    print("type     :", type_name)

    print("\nAttributes:")
    for attr in prim.GetAttributes():
        aname = attr.GetName()
        try:
            value = attr.Get()
        except Exception as e:
            value = f"<read failed: {e}>"
        print(f"  {aname}: {value}")

    print("\nRelationships:")
    for rel in prim.GetRelationships():
        rname = rel.GetName()
        try:
            targets = rel.GetTargets()
        except Exception as e:
            targets = f"<read failed: {e}>"
        print(f"  {rname}: {targets}")


print("\n\n==================== IMPORTANT LINKS / EE ====================")

for prim in stage.TraverseAll():
    name = prim.GetName()
    path = str(prim.GetPath())
    type_name = prim.GetTypeName()

    lower = (name + " " + path).lower()

    if not any(k in lower for k in EE_NAMES):
        continue

    print("\n" + "=" * 100)
    print("prim path:", path)
    print("name     :", name)
    print("type     :", type_name)

    for attr in prim.GetAttributes():
        aname = attr.GetName()
        lname = aname.lower()

        if (
            "xformop" in lname
            or "translate" in lname
            or "orient" in lname
            or "rotate" in lname
            or "physics" in lname
            or "mass" in lname
        ):
            try:
                value = attr.Get()
            except Exception as e:
                value = f"<read failed: {e}>"
            print(f"  {aname}: {value}")

    for rel in prim.GetRelationships():
        rname = rel.GetName()
        try:
            targets = rel.GetTargets()
        except Exception as e:
            targets = f"<read failed: {e}>"
        print(f"  REL {rname}: {targets}")


print("\n\n==================== SUMMARY CANDIDATE JOINT DATA ====================")

for prim in stage.TraverseAll():
    name = prim.GetName()
    type_name = prim.GetTypeName()

    if name not in JOINT_NAMES:
        continue

    print("\n" + "-" * 80)
    print("name:", name)
    print("path:", prim.GetPath())
    print("type:", type_name)

    keys = [
        "physics:axis",
        "physics:localPos0",
        "physics:localRot0",
        "physics:localPos1",
        "physics:localRot1",
        "physics:lowerLimit",
        "physics:upperLimit",
        "drive:angular:physics:targetPosition",
        "drive:angular:physics:stiffness",
        "drive:angular:physics:damping",
    ]

    for key in keys:
        attr = prim.GetAttribute(key)
        if attr:
            try:
                print(f"{key}: {attr.Get()}")
            except Exception as e:
                print(f"{key}: <read failed {e}>")

    for rel_name in ["physics:body0", "physics:body1"]:
        rel = prim.GetRelationship(rel_name)
        if rel:
            try:
                print(f"{rel_name}: {rel.GetTargets()}")
            except Exception as e:
                print(f"{rel_name}: <read failed {e}>")
