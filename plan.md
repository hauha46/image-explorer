[single image + optional prompt + λ + neutral_prompt]
│   (HTTP POST /process)
▼
[DepthPro]  →  metric depth map D(u,v)  +  horizontal FOV
│
▼
[Build K from FOV]  →  intrinsics K  +  initial c2w = I
│
▼
[NVS backend:  SEVA  or  ViewCrafter]  →  10 novel-view PNGs
│
▼
[DUSt3R]  →  scene.glb  (colored point cloud, global alignment)
│
▼
[Ball-pivoting mesher]  →  scene_mesh.glb  (watertight-ish mesh)
│
▼
[Scene composer]  →  scene.json  (manifest: views + cloud + mesh + camera)
│
▼
[Three.js frontend viewer]  →  interactive 3D scene in browser