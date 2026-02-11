import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── Config ──────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const MOVE_SPEED = 4.0;
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

        // 2. Create Object Layers (Billboarded Depth Meshes OR 3D Models)
        if (sceneData.objects) {
            for (const obj of sceneData.objects) {
                if (obj.type === 'model') {
                    await createModelLayer(
                        `${API_BASE}${obj.url}`,
                        obj.position_uv,
                        obj.depth_val,
                        obj.bbox_uv,
                        cam.fov, cam.near, cam.far,
                        sceneData.metadata.image_size
                    );
                } else {
                    await createObjectLayer(
                        `${API_BASE}${obj.url}`,
                        `${API_BASE}${obj.depth_url}`,
                        obj.position_uv,
                        obj.depth_val,
                        obj.bbox_uv,
                        cam.fov, cam.near, cam.far,
                        sceneData.metadata.image_size
                    );
                }
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

    // Wrap angle: 100 degrees
    const hFov = THREE.MathUtils.degToRad(100);
    // Vertical FOV matched to aspect
    const vFov = hFov / (depthImg.width / depthImg.height);

    for (let i = 0; i < pos.count; i++) {
        const u = uv.getX(i);
        const v = uv.getY(i);

        // Sample depth
        const px = Math.floor(u * (canvas.width - 1));
        const py = Math.floor((1 - v) * (canvas.height - 1));
        const idx = (py * canvas.width + px) * 4;
        const d_raw = data[idx] / 255.0;
        // Invert: 1.0 (Near) -> 0.0 -> Radius=Near
        //         0.0 (Far)  -> 1.0 -> Radius=Far
        const d_inv = 1.0 - d_raw;

        const radius = near + (d_inv * (far - near));

        // Cylindrical mapping
        const phi = (u - 0.5) * hFov;
        const y_ang = (v - 0.5) * vFov;
        const height = radius * Math.tan(y_ang); // Approximate height on cylinder surface

        const x = radius * Math.sin(phi);
        const y = height;
        const z = -radius * Math.cos(phi);

        pos.setXYZ(i, x, y, z);
    }

    geometry.computeVertexNormals();
    return geometry;
}

/**
 * Creates a standalone object layer positioned in 3D space.
 */
function createObjectLayer(imageUrl, depthUrl, posUv, depthVal, bboxUv, fovDeg, near, far, imgSize) {
    return new Promise((resolve) => {
        new THREE.TextureLoader().load(imageUrl, (colorTex) => {
            colorTex.colorSpace = THREE.SRGBColorSpace;
            new THREE.TextureLoader().load(depthUrl, (depthTex) => {

                // Calculate position in 3D
                // Calculate position in 3D
                const z_dist = near + depthVal * (far - near);

                // Position helpers
                const hFov = THREE.MathUtils.degToRad(100);
                const aspect = imgSize[0] / imgSize[1];
                const vFov = hFov / aspect;

                // Cylindrical angle of object center
                const phi = (posUv[0] - 0.5) * hFov;
                const theta = (posUv[1] - 0.5) * vFov;

                // Object center position matching background projection
                const cx = z_dist * Math.sin(phi);
                const cy = z_dist * Math.tan(theta);
                const cz = -z_dist * Math.cos(phi);

                // Scale calculation (PHYSICALLY ACCURATE)
                // We want the object's 3D size to match its 2D visual size at that depth.

                // Vertical size:
                // Total visible vertical angle is vFov.
                // In cylindrical projection, Y is height = r * tan(theta).
                // The visible height range is roughly [ r*tan(-vFov/2), r*tan(vFov/2) ]
                // Total height H = 2 * r * tan(vFov/2)
                const totalH = 2 * z_dist * Math.tan(vFov / 2);

                // Object height in world units
                const bboxH_uv = bboxUv[3] - bboxUv[1];
                const worldH = totalH * bboxH_uv;

                // Object width in world units (preserve aspect ratio of the crop)
                const cropW = colorTex.image.width;
                const cropH = colorTex.image.height;
                const worldW = worldH * (cropW / cropH);

                // Create depth-displaced plane for the object
                const geometry = createObjectDepthGeometry(depthTex.image, 1.0); // 1.0 depth scale
                const material = new THREE.MeshBasicMaterial({
                    map: colorTex,
                    transparent: true,
                    side: THREE.DoubleSide,
                    alphaTest: 0.1
                });

                const mesh = new THREE.Mesh(geometry, material);

                mesh.scale.set(worldW, worldH, 1);

                mesh.position.set(cx, cy, cz);
                mesh.position.set(cx, cy, cz);
                mesh.lookAt(0, cy, 0); // Face the vertical axis

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

/**
 * Creates a true 3D model layer from GLB.
 */
function createModelLayer(modelUrl, posUv, depthVal, bboxUv, fovDeg, near, far, imgSize) {
    return new Promise((resolve) => {
        const loader = new GLTFLoader();
        loader.load(modelUrl, (gltf) => {
            const mesh = gltf.scene;

            // Calculate position (Same logic as object layer)
            const z_dist = near + depthVal * (far - near);

            // Position helpers
            const hFov = THREE.MathUtils.degToRad(100);
            const aspect = imgSize[0] / imgSize[1];
            const vFov = hFov / aspect;

            // Cylindrical angle
            const phi = (posUv[0] - 0.5) * hFov;
            const theta = (posUv[1] - 0.5) * vFov;

            // Cartesian coordinates
            const cx = z_dist * Math.sin(phi);
            const cy = z_dist * Math.tan(theta);
            const cz = -z_dist * Math.cos(phi);

            // Calculate Target Scale (Physical World Size)
            const totalH = 2 * z_dist * Math.tan(vFov / 2);
            const bboxH_uv = bboxUv[3] - bboxUv[1];
            const worldH = totalH * bboxH_uv;

            // Measure Model
            const box = new THREE.Box3().setFromObject(mesh);
            const modelH = box.max.y - box.min.y || 1.0;

            // Apply scale
            const scale = worldH / modelH;
            mesh.scale.set(scale, scale, scale);

            // Position
            mesh.position.set(cx, cy, cz);

            // Orient: Look at center line
            mesh.lookAt(0, cy, 0);

            scene.add(mesh);
            resolve();
        }, undefined, (err) => {
            console.error("Failed to load GLB:", err);
            resolve(); // Don't block scene load
        });
    });
}