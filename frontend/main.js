import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// ── Config ──────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const MOVE_SPEED = 1.0;
const SPRINT_MULTIPLIER = 2.0;

// ── State ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let session_id = null;

const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
const keys = { forward: false, backward: false, left: false, right: false, up: false, down: false, sprint: false };
let prevTime = performance.now();

// ── DOM ─────────────────────────────────────────────────────────────
const blocker = document.getElementById('blocker');
const info = document.getElementById('info');
const fileInput = document.getElementById('file-upload');
const uploadBtn = document.getElementById('upload-btn');
const progressBar = document.querySelector('.progress-fill');
const progressText = document.querySelector('.progress-text');
const progressBarContainer = document.querySelector('.progress-bar');
const crosshair = document.getElementById('crosshair');

let sceneLoaded = false;

init();
animate();

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Initial camera position at origin
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 500);
    camera.position.set(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.body.appendChild(renderer.domElement);

    // Unlit scene (textures carry the lighting)
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(ambientLight);

    // Controls
    controls = new PointerLockControls(camera, document.body);
    setupControls();

    scene.add(controls.object);

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);
    uploadBtn.addEventListener('click', uploadImage);
}

function setupControls() {
    controls.addEventListener('lock', () => {
        if (sceneLoaded) {
            blocker.style.display = 'none';
            if (crosshair) crosshair.style.display = 'block';
        }
    });

    controls.addEventListener('unlock', () => {
        if (sceneLoaded) {
            blocker.style.display = 'flex';
            blocker.innerHTML = `
                <div class="upload-container">
                    <h1>Paused</h1>
                    <p>Click to resume</p>
                    <p class="controls-hint">WASD — Move &nbsp;|&nbsp; Mouse — Look &nbsp;|&nbsp; Shift — Sprint</p>
                </div>
            `;
            if (crosshair) crosshair.style.display = 'none';
        }
    });

    blocker.addEventListener('click', (e) => {
        if (sceneLoaded && !e.target.closest('button') && !e.target.closest('input')) {
            controls.lock();
        }
    });
}

function onKeyDown(event) {
    switch (event.code) {
        case 'KeyW': case 'ArrowUp': keys.forward = true; break;
        case 'KeyS': case 'ArrowDown': keys.backward = true; break;
        case 'KeyA': case 'ArrowLeft': keys.left = true; break;
        case 'KeyD': case 'ArrowRight': keys.right = true; break;
        case 'Space': keys.up = true; event.preventDefault(); break;
        case 'ControlLeft': case 'ControlRight': keys.down = true; break;
        case 'ShiftLeft': case 'ShiftRight': keys.sprint = true; break;
    }
}

function onKeyUp(event) {
    switch (event.code) {
        case 'KeyW': case 'ArrowUp': keys.forward = false; break;
        case 'KeyS': case 'ArrowDown': keys.backward = false; break;
        case 'KeyA': case 'ArrowLeft': keys.left = false; break;
        case 'KeyD': case 'ArrowRight': keys.right = false; break;
        case 'Space': keys.up = false; break;
        case 'ControlLeft': case 'ControlRight': keys.down = false; break;
        case 'ShiftLeft': case 'ShiftRight': keys.sprint = false; break;
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    const time = performance.now();
    const delta = Math.min((time - prevTime) / 1000, 0.1);
    prevTime = time;

    if (controls.isLocked) {
        velocity.x -= velocity.x * 8.0 * delta;
        velocity.z -= velocity.z * 8.0 * delta;
        velocity.y -= velocity.y * 8.0 * delta;

        direction.z = Number(keys.forward) - Number(keys.backward);
        direction.x = Number(keys.right) - Number(keys.left);
        direction.y = Number(keys.up) - Number(keys.down);
        direction.normalize();

        const speed = MOVE_SPEED * (keys.sprint ? SPRINT_MULTIPLIER : 1.0);

        if (keys.forward || keys.backward) velocity.z -= direction.z * speed * delta;
        if (keys.left || keys.right) velocity.x += direction.x * speed * delta;
        if (keys.up || keys.down) velocity.y += direction.y * speed * delta;

        controls.moveRight(velocity.x);
        controls.moveForward(-velocity.z);
        camera.position.y += velocity.y;
    }

    renderer.render(scene, camera);
}

// ── Upload ──────────────────────────────────────────────────────────
async function uploadImage() {
    if (!fileInput.files.length) return alert('Please select an image');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    if (progressBarContainer) progressBarContainer.style.display = 'block';
    uploadBtn.disabled = true;
    uploadBtn.innerText = 'Processing...';
    info.style.display = 'block';

    try {
        const res = await fetch(`${API_BASE}/process`, { method: 'POST', body: formData });
        const data = await res.json();
        session_id = data.session_id;
        pollStatus();
    } catch (e) {
        console.error(e);
        alert('Upload failed: ' + e.message);
        uploadBtn.disabled = false;
        uploadBtn.innerText = 'Generate Scene';
    }
}

async function pollStatus() {
    if (!session_id) return;
    try {
        const res = await fetch(`${API_BASE}/status/${session_id}`);
        const status = await res.json();
        if (progressBar) progressBar.style.width = status.progress + '%';
        if (progressText) progressText.innerText = status.current_step || '';
        info.innerText = status.current_step || 'Processing...';

        if (status.status === 'complete') loadScene(session_id);
        else if (status.status === 'error') { alert('Error: ' + status.current_step); location.reload(); }
        else setTimeout(pollStatus, 1000);
    } catch (e) {
        console.error("Poll failed", e);
        setTimeout(pollStatus, 2000);
    }
}

// ── Scene Loading ───────────────────────────────────────────────────
async function loadScene(sid) {
    info.innerText = 'Building Layered Scene...';

    try {
        const res = await fetch(`${API_BASE}/uploads/${sid}/scene.json`);
        const sceneData = await res.json();
        console.log('Scene data:', sceneData);

        const cam = sceneData.camera;

        // 1. Create Background (Cylindrical Projection)
        if (sceneData.background) {
            const bg = sceneData.background;
            await createBackgroundLayer(
                `${API_BASE}${bg.url}`,
                `${API_BASE}${bg.depth_url}`,
                cam.fov, cam.near, cam.far
            );
        }

        // 2. Create Object Layers (2.5D Depth Meshes)
        if (sceneData.objects) {
            for (const obj of sceneData.objects) {
                await createObjectLayer(
                    `${API_BASE}${obj.url}`,
                    `${API_BASE}${obj.depth_url}`,
                    obj.position_uv,
                    obj.bottom_uv || obj.position_uv,
                    obj.depth_val,
                    obj.bbox_uv,
                    cam.fov, cam.near, cam.far,
                    sceneData.metadata.image_size
                );
            }
        }

        // Ready
        sceneLoaded = true;
        info.style.display = 'none';
        blocker.innerHTML = `
            <div class="upload-container">
                <h1>Scene Ready!</h1>
                <p>Click to explore</p>
                <p class="controls-hint">WASD — Move &nbsp;|&nbsp; Mouse — Look &nbsp;|&nbsp; Shift — Sprint</p>
            </div>
        `;
        blocker.style.display = 'flex';

    } catch (e) {
        console.error("Scene load failed", e);
        alert("Could not load scene: " + e.message);
    }
}

/**
 * Creates the immersive background layer using cylindrical projection.
 */
function createBackgroundLayer(imageUrl, depthUrl, fov, near, far) {
    return new Promise((resolve) => {
        new THREE.TextureLoader().load(imageUrl, (colorTex) => {
            colorTex.colorSpace = THREE.SRGBColorSpace;
            new THREE.TextureLoader().load(depthUrl, (depthTex) => {
                const geometry = createCylindricalDepthGeometry(
                    depthTex.image, fov, 1.0, near, far
                );
                const material = new THREE.MeshBasicMaterial({ map: colorTex, side: THREE.DoubleSide });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.name = "background";
                scene.add(mesh);
                resolve();
            });
        });
    });
}

function createCylindricalDepthGeometry(depthImg, fovDeg, aspect, near, far) {
    const canvas = document.createElement('canvas');
    canvas.width = depthImg.width;
    canvas.height = depthImg.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(depthImg, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    const segW = 300;
    const segH = 200;
    const geometry = new THREE.PlaneGeometry(1, 1, segW, segH);
    const pos = geometry.attributes.position;
    const uv = geometry.attributes.uv;

    // Use the scene's actual FOV — NOT a hardcoded value
    const hFov = THREE.MathUtils.degToRad(fovDeg);
    // Vertical FOV from aspect ratio
    const vFov = hFov * (depthImg.height / depthImg.width);

    // First pass: position all vertices
    // Store per-vertex radius for depth-tear detection
    const radii = new Float32Array(pos.count);

    for (let i = 0; i < pos.count; i++) {
        const u = uv.getX(i);
        const v = uv.getY(i);

        // Sample depth
        const px = Math.floor(u * (canvas.width - 1));
        const py = Math.floor((1 - v) * (canvas.height - 1));
        const idx = (py * canvas.width + px) * 4;
        const d_raw = data[idx] / 255.0;
        const d_inv = 1.0 - d_raw;

        const radius = near + (d_inv * (far - near));
        radii[i] = radius;

        // Cylindrical mapping
        const phi = (u - 0.5) * hFov;
        const y_ang = (v - 0.5) * vFov;
        const height = radius * Math.tan(y_ang);

        const x = radius * Math.sin(phi);
        const y = height;
        const z = -radius * Math.cos(phi);

        pos.setXYZ(i, x, y, z);
    }

    // Second pass: degenerate triangles at depth discontinuities
    // This prevents rubber-band stretching where depth changes sharply
    const index = geometry.index;
    if (index) {
        const indices = index.array;
        const depthRange = far - near;
        const tearThreshold = depthRange * 0.15; // 15% of total range

        for (let t = 0; t < indices.length; t += 3) {
            const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
            const r0 = radii[i0], r1 = radii[i1], r2 = radii[i2];
            const maxR = Math.max(r0, r1, r2);
            const minR = Math.min(r0, r1, r2);

            if (maxR - minR > tearThreshold) {
                // Collapse triangle to a degenerate point
                indices[t] = i0;
                indices[t + 1] = i0;
                indices[t + 2] = i0;
            }
        }
        index.needsUpdate = true;
    }

    geometry.computeVertexNormals();
    return geometry;
}

/**
 * Creates a standalone object layer positioned in 3D space.
 * Uses bottom-anchor UV for grounding.
 */
function createObjectLayer(imageUrl, depthUrl, posUv, bottomUv, depthVal, bboxUv, fovDeg, near, far, imgSize) {
    return new Promise((resolve) => {
        new THREE.TextureLoader().load(imageUrl, (colorTex) => {
            colorTex.colorSpace = THREE.SRGBColorSpace;
            new THREE.TextureLoader().load(depthUrl, (depthTex) => {

                // Calculate Z distance from depth
                const z_dist = near + depthVal * (far - near);

                // Use scene FOV — NOT hardcoded
                const hFov = THREE.MathUtils.degToRad(fovDeg);
                const aspect = imgSize[0] / imgSize[1];
                const vFov = hFov * (imgSize[1] / imgSize[0]);

                // Use BOTTOM UV for Y anchor (so objects sit on the ground)
                const phi = (bottomUv[0] - 0.5) * hFov;
                const theta_bottom = -(bottomUv[1] - 0.5) * vFov;

                // Bottom anchor position
                const bx = z_dist * Math.sin(phi);
                const by = z_dist * Math.tan(theta_bottom);
                const bz = -z_dist * Math.cos(phi);

                // Scale calculation
                const totalH = 2 * z_dist * Math.tan(vFov / 2);
                const bboxH_uv = bboxUv[3] - bboxUv[1];
                const worldH = totalH * bboxH_uv;
                const cropW = colorTex.image.width;
                const cropH = colorTex.image.height;
                const worldW = worldH * (cropW / cropH);

                // Create depth-displaced plane
                const geometry = createObjectDepthGeometry(depthTex.image, 1.0);
                const material = new THREE.MeshBasicMaterial({
                    map: colorTex,
                    transparent: true,
                    side: THREE.DoubleSide,
                    alphaTest: 0.1
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.scale.set(worldW, worldH, 1);

                // Position: anchor at bottom, shift up by half height
                mesh.position.set(bx, by + worldH / 2, bz);
                mesh.lookAt(0, by + worldH / 2, 0);

                scene.add(mesh);
                resolve();
            });
        });
    });
}

function createObjectDepthGeometry(depthImg, depthScale) {
    const canvas = document.createElement('canvas');
    canvas.width = depthImg.width;
    canvas.height = depthImg.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(depthImg, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    const segW = 64;
    const segH = 64;
    const geometry = new THREE.PlaneGeometry(1, 1, segW, segH);
    const pos = geometry.attributes.position;
    const uv = geometry.attributes.uv;

    // Calculate mean depth to center the object
    let meanDepth = 0;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
        if (data[i] > 10) { // ignore background black
            meanDepth += data[i];
            count++;
        }
    }
    meanDepth = (count > 0 ? meanDepth / count : 128) / 255.0;

    for (let i = 0; i < pos.count; i++) {
        const u = uv.getX(i);
        const v = uv.getY(i);

        const px = Math.floor(u * (canvas.width - 1));
        const py = Math.floor((1 - v) * (canvas.height - 1));

        const idx = (py * canvas.width + px) * 4;
        const d_norm = data[idx] / 255.0;

        // Push verts out based on local depth variation
        // We subtract meanDepth so the object stays roughly at its anchor point
        const z = (d_norm - meanDepth) * depthScale;

        pos.setZ(i, z);
    }

    geometry.computeVertexNormals();
    return geometry;
}