import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── Config ──────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const MOVE_SPEED = 0.05;    // Reduced base speed for better point cloud exploration
const SPRINT_MULTIPLIER = 2.0;

// ── State ───────────────────────────────────────────────────────────
let scene, camera, renderer, controls;

const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
const keys = { forward: false, backward: false, left: false, right: false, up: false, down: false, sprint: false };
let prevTime = performance.now();

// ── DOM ─────────────────────────────────────────────────────────────
const blocker = document.getElementById('blocker');
const info = document.getElementById('info');
const loadBtn = document.getElementById('load-test-btn');
const crosshair = document.getElementById('crosshair');

let sceneLoaded = false;

init();
animate();

function init() {
    scene = new THREE.Scene();
    // Dust3r puts black background, let's keep it dark to see the points better
    scene.background = new THREE.Color(0x111111);

    // Initial camera position at origin
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    // GLB colors often look best with standard sRGB output
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    document.body.appendChild(renderer.domElement);

    // Add ambient light to ensure vertex colors/materials are visible if needed
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
    scene.add(ambientLight);

    // Controls
    controls = new PointerLockControls(camera, document.body);
    setupControls();

    scene.add(controls.object);

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);

    // Bind the new load test button
    loadBtn.addEventListener('click', loadTestScene);
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
                    <p class="controls-hint">WASD — Move &nbsp;|&nbsp; Mouse — Look &nbsp;|&nbsp; E/Q — Up/Down &nbsp;|&nbsp; Shift — Sprint</p>
                </div>
            `;
            if (crosshair) crosshair.style.display = 'none';
        }
    });

    blocker.addEventListener('click', (e) => {
        if (sceneLoaded && !e.target.closest('button')) {
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
        case 'KeyE': case 'Space': keys.up = true; event.preventDefault(); break;
        case 'KeyQ': case 'ControlLeft': keys.down = true; event.preventDefault(); break;
        case 'ShiftLeft': case 'ShiftRight': keys.sprint = true; break;
    }
}

function onKeyUp(event) {
    switch (event.code) {
        case 'KeyW': case 'ArrowUp': keys.forward = false; break;
        case 'KeyS': case 'ArrowDown': keys.backward = false; break;
        case 'KeyA': case 'ArrowLeft': keys.left = false; break;
        case 'KeyD': case 'ArrowRight': keys.right = false; break;
        case 'KeyE': case 'Space': keys.up = false; break;
        case 'KeyQ': case 'ControlLeft': keys.down = false; break;
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
        // Damping
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
        // Vertical movement relative to world, not camera
        camera.position.y += velocity.y;
    }

    renderer.render(scene, camera);
}

// ── Test Scene Loading ──────────────────────────────────────────────
function loadTestScene() {
    loadBtn.disabled = true;
    loadBtn.innerText = 'Downloading GLB from Server...';
    info.style.display = 'block';

    const loader = new GLTFLoader();

    // We expect the backend to have the mesh.glb available at this path
    const glbUrl = `${API_BASE}/uploads/test_scene/scene.glb`;

    loader.load(
        glbUrl,
        function (gltf) {
            info.innerText = 'GLB Loaded! Initializing environment...';

            const model = gltf.scene;

            // Log the model to see what Dust3r generated (meshes vs points)
            console.log("Loaded GLTF Model:", model);

            // Force the material to show vertex colors and render both sides!
            // Poisson reconstruction often puts us "inside" the mesh, meaning standard backface culling makes it transparent.
            model.traverse((child) => {
                if (child.isMesh) {
                    // Replace generic materials with an unlit Basic material to strictly show the vertex colors
                    child.material = new THREE.MeshBasicMaterial({
                        vertexColors: true,
                        side: THREE.DoubleSide
                    });
                }
            });

            // 1. Flip the model to correct coordinate system mismatch
            // Dust3r uses OpenCV (Y-down), Three.js uses OpenGL (Y-up)
            // model.rotation.x = Math.PI;

            // Center the model after rotation
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());

            model.position.x += (model.position.x - center.x);
            model.position.y += (model.position.y - center.y);
            model.position.z += (model.position.z - center.z);

            // model.position.x -= center.x;
            // model.position.y -= center.y;
            // model.position.z -= center.z;

            scene.add(model);

            // Position camera slightly back so we can see it
            camera.position.set(0, 0, 2);

            sceneLoaded = true;
            controls.lock(); // Try to auto-lock instructions screen
            info.innerText = 'W/A/S/D to move. Q/E to fly. Shift to sprint.';

            setTimeout(() => {
                info.style.display = 'none';
            }, 5000);
        },
        function (xhr) {
            const percent = (xhr.loaded / xhr.total * 100);
            if (percent) {
                loadBtn.innerText = `Downloading: ${Math.round(percent)}%`;
            }
        },
        function (error) {
            console.error('Error loading test scene GLB:', error);
            alert('Failed to load test scene. Was it generated successfully in backend/uploads/test_scene/scene.glb?');
            loadBtn.disabled = false;
            loadBtn.innerText = 'Load Test Scene';
        }
    );
}