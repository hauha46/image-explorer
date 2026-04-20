---
name: Fix 3D Viewer Zoom
overview: Fix the "super zoomed out" 3D viewer by auto-framing the camera to the model's bounding box, and add OrbitControls as the default 3D navigation with a toggle to switch into FPS walk-around mode.
todos:
  - id: auto-frame
    content: Update loadGLB to compute bounding sphere and auto-position camera + scale MOVE_SPEED
    status: completed
  - id: orbit-controls
    content: Import OrbitControls, create instance in init(), wire up enable/disable logic
    status: completed
  - id: mode-bar-ui
    content: Add FPS Walk button to index.html mode bar and update switchMode() for 3 modes
    status: completed
  - id: verify-lints
    content: Check for linter errors in edited files
    status: completed
isProject: false
---

# Fix 3D Viewer: Auto-Framing + Dual Control Modes

## Problem

The `loadGLB` function in [frontend/main.js](frontend/main.js) hardcodes `camera.position.set(0, 0, 2)` after loading, regardless of the model's actual size. DUSt3R point clouds can vary wildly in scale, so the camera often ends up far too close or too far from the scene. Additionally, the only 3D navigation is FPS PointerLock with a fixed `MOVE_SPEED = 0.05`, making it impossible to comfortably frame the model.

## Solution

### 1. Auto-frame camera to bounding box (in `loadGLB`)

After loading the GLB model, compute its bounding box and bounding sphere. Position the camera so the entire model is visible:

```javascript
const box = new THREE.Box3().setFromObject(model);
const sphere = box.getBoundingSphere(new THREE.Sphere());
const fov = camera.fov * (Math.PI / 180);
const dist = sphere.radius / Math.sin(fov / 2);
camera.position.copy(sphere.center);
camera.position.z += dist * 1.3;
camera.lookAt(sphere.center);
```

Also scale `MOVE_SPEED` proportionally to the scene size so FPS movement feels natural regardless of scale.

### 2. Add OrbitControls as default 3D mode

Import `OrbitControls` from Three.js addons. When the user clicks "3D Explore", initialize OrbitControls targeting the model center (from the bounding sphere). This gives standard click-drag rotation and scroll-to-zoom out of the box.

Set `orbitControls.target` to `sphere.center` so the user orbits around the actual model, not the world origin.

### 3. Toggle between Orbit and FPS modes

Add a third button to the mode bar: **"FPS Walk"**. The three viewer modes become:

- **Orbit Frames** -- existing 2D frame scrubber (unchanged)
- **3D Explore** -- OrbitControls (rotate/zoom around model)
- **FPS Walk** -- existing PointerLock FPS controls

When switching between 3D Explore and FPS Walk, transfer the current camera position so the transition is seamless.

### Files Changed

- **[frontend/main.js](frontend/main.js)**: All logic changes (OrbitControls import, auto-framing in `loadGLB`, mode switching, dynamic MOVE_SPEED)
- **[frontend/index.html](frontend/index.html)**: Add the third "FPS Walk" button to the mode bar

### Key Details

- `OrbitControls` is available at `three/addons/controls/OrbitControls.js` (already in the Three.js CDN importmap)
- The existing `PointerLockControls` stay for FPS mode; `OrbitControls` is created alongside it
- Only one set of controls is active at a time (disable the other via `.enabled = false`)
- The `animate()` loop calls `orbitControls.update()` when in orbit-3d mode
- Scene center and radius are stored after `loadGLB` for use across mode switches
