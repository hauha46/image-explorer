import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ── Config ──────────────────────────────────────────────────────────
const API_BASE = window.location.origin;
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
const uploadPanel = document.getElementById('upload-panel');
const info = document.getElementById('info');
const loadBtn = document.getElementById('load-test-btn');
const crosshair = document.getElementById('crosshair');
const fileInput = document.getElementById('file-input');
const modelSelect = document.getElementById('model-select');
const uploadBtn = document.getElementById('upload-btn');
const progressBar = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const modeBar = document.getElementById('mode-bar');
const modeOrbitBtn = document.getElementById('mode-orbit-btn');
const mode3dBtn = document.getElementById('mode-3d-btn');
const orbitViewer = document.getElementById('orbit-viewer');
const orbitCanvas = document.getElementById('orbit-canvas');
const orbitFrameCounter = document.getElementById('orbit-frame-counter');
const promptRow = document.getElementById('prompt-row');
const promptInput = document.getElementById('prompt-input');
const newSceneBtn = document.getElementById('new-scene-btn');

const PROMPT_MODELS = ['viewcrafter', 'panodreamer'];

let sceneLoaded = false;
let currentSessionId = null;
let currentMode = 'orbit'; // 'orbit' or '3d'

// ── Orbit Viewer State ──────────────────────────────────────────────
const orbitState = {
    images: [],
    loaded: false,
    currentIndex: 0,
    isDragging: false,
    startX: 0,
    startIndex: 0,
    ctx: null,
};

init();
animate();

function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    renderer.outputColorSpace = THREE.SRGBColorSpace;

    document.body.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
    scene.add(ambientLight);

    controls = new PointerLockControls(camera, document.body);
    setupControls();

    scene.add(controls.object);

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    window.addEventListener('resize', onWindowResize);

    loadBtn.addEventListener('click', loadTestScene);
    fileInput.addEventListener('change', () => {
        uploadBtn.disabled = !fileInput.files.length;
    });
    uploadBtn.addEventListener('click', handleUpload);

    // Show/hide prompt row when model changes
    modelSelect.addEventListener('change', () => {
        promptRow.style.display = PROMPT_MODELS.includes(modelSelect.value) ? 'flex' : 'none';
    });

    // Mode toggle
    modeOrbitBtn.addEventListener('click', () => switchMode('orbit'));
    mode3dBtn.addEventListener('click', () => switchMode('3d'));
    newSceneBtn.addEventListener('click', resetToUpload);

    // Orbit viewer mouse/touch events
    orbitViewer.addEventListener('mousedown', onOrbitMouseDown);
    window.addEventListener('mousemove', onOrbitMouseMove);
    window.addEventListener('mouseup', onOrbitMouseUp);
    orbitViewer.addEventListener('touchstart', onOrbitTouchStart, { passive: false });
    window.addEventListener('touchmove', onOrbitTouchMove, { passive: false });
    window.addEventListener('touchend', onOrbitTouchEnd);

    orbitState.ctx = orbitCanvas.getContext('2d');
}

// ── Upload & Pipeline ───────────────────────────────────────────────
async function handleUpload() {
    const file = fileInput.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadBtn.innerText = 'Uploading…';
    progressBar.style.display = 'block';
    progressFill.style.width = '5%';
    progressText.innerText = 'Uploading image…';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelSelect.value);
    const prompt = promptInput.value.trim();
    if (prompt) formData.append('prompt', prompt);

    try {
        const res = await fetch(`${API_BASE}/process`, { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        currentSessionId = data.session_id;
        uploadBtn.innerText = 'Processing…';
        pollStatus(currentSessionId);
    } catch (err) {
        alert(`Error: ${err.message}`);
        resetUploadUI();
    }
}

async function pollStatus(sessionId) {
    const poll = async () => {
        try {
            const res = await fetch(`${API_BASE}/status/${sessionId}`);
            const data = await res.json();

            if (data.status === 'processing') {
                progressFill.style.width = `${data.progress}%`;
                progressText.innerText = data.current_step || 'Processing…';
                setTimeout(poll, 1500);
            } else if (data.status === 'complete') {
                progressFill.style.width = '100%';
                progressText.innerText = 'Done! Loading results…';
                await loadSessionResults(sessionId);
            } else if (data.status === 'error') {
                throw new Error(data.current_step || 'Pipeline failed');
            } else {
                setTimeout(poll, 2000);
            }
        } catch (err) {
            alert(`Pipeline error: ${err.message}`);
            resetUploadUI();
        }
    };
    poll();
}

async function loadSessionResults(sessionId) {
    // Load orbit frames
    try {
        const res = await fetch(`${API_BASE}/views/${sessionId}`);
        const data = await res.json();
        if (data.views && data.views.length > 0) {
            await loadOrbitImages(data.views);
        }
    } catch (e) {
        console.warn('Could not load orbit views:', e);
    }

    // Load GLB for 3D mode
    loadGLB(`${API_BASE}/uploads/${sessionId}/scene.glb`);
}

function resetUploadUI() {
    uploadBtn.disabled = false;
    uploadBtn.innerText = 'Generate Scene';
    progressBar.style.display = 'none';
    progressFill.style.width = '0%';
    progressText.innerText = '';
}

function resetToUpload() {
    // Clear 3D scene
    const toRemove = [];
    scene.traverse((child) => {
        if (child.isMesh || child.isPoints) toRemove.push(child);
    });
    toRemove.forEach(obj => {
        obj.geometry?.dispose();
        obj.material?.dispose();
        obj.parent?.remove(obj);
    });
    camera.position.set(0, 0, 0);

    // Clear orbit viewer state
    orbitState.images = [];
    orbitState.loaded = false;
    orbitState.currentIndex = 0;

    // Hide viewers
    orbitViewer.style.display = 'none';
    orbitFrameCounter.style.display = 'none';
    renderer.domElement.style.display = 'none';
    modeBar.style.display = 'none';
    if (crosshair) crosshair.style.display = 'none';
    if (controls.isLocked) controls.unlock();

    // Restore upload panel
    sceneLoaded = false;
    currentSessionId = null;
    uploadPanel.innerHTML = `
      <h1>AI 3D Scene Reconstructor</h1>
      <p>Upload an image to generate novel views and a 3D point cloud.</p>

      <input type="file" id="file-input" accept="image/*">

      <div class="form-row">
        <label for="model-select">Model:</label>
        <select id="model-select">
          <option value="svd">SVD \u2014 Video Diffusion (fast, general)</option>
          <option value="viewcrafter">ViewCrafter \u2014 Point Cloud Guided (text prompt)</option>
          <option value="seva">SEVA \u2014 Stable Virtual Camera (orbit, best quality)</option>
          <option value="sv3d">SV3D \u2014 Stable Video 3D (orbital, object-centric)</option>
          <option value="zero123pp">Zero123++ \u2014 Multi-View Grid (6 views)</option>
          <option value="vivid">VIVID \u2014 Video Diffusion</option>
          <option value="panodreamer">PanoDreamer \u2014 Panoramic (text prompt)</option>
        </select>
      </div>

      <div class="form-row" id="prompt-row" style="display: none;">
        <label for="prompt-input">Prompt:</label>
        <input type="text" id="prompt-input" placeholder="e.g. Rotating view of a building">
      </div>

      <button id="upload-btn" disabled>Generate Scene</button>

      <div style="margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 16px;">
        <p style="font-size: 12px; color: rgba(255,255,255,0.35);">Or load a pre-generated test scene:</p>
        <button id="load-test-btn" class="secondary" style="margin-top: 8px;">Load Test Scene</button>
      </div>

      <div class="progress-bar" id="progress-bar">
        <div class="progress-fill" id="progress-fill"></div>
      </div>
      <div class="progress-text" id="progress-text"></div>
    `;
    blocker.style.display = 'flex';

    // Re-bind event listeners to the new DOM elements
    rebindUploadListeners();
}

function rebindUploadListeners() {
    const fi = document.getElementById('file-input');
    const ub = document.getElementById('upload-btn');
    const ms = document.getElementById('model-select');
    const lb = document.getElementById('load-test-btn');
    const pr = document.getElementById('prompt-row');
    const pi = document.getElementById('prompt-input');

    fi.addEventListener('change', () => { ub.disabled = !fi.files.length; });
    ub.addEventListener('click', () => {
        const file = fi.files[0];
        if (!file) return;
        ub.disabled = true;
        ub.innerText = 'Uploading\u2026';
        const pb = document.getElementById('progress-bar');
        const pf = document.getElementById('progress-fill');
        const pt = document.getElementById('progress-text');
        pb.style.display = 'block';
        pf.style.width = '5%';
        pt.innerText = 'Uploading image\u2026';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', ms.value);
        const prompt = pi.value.trim();
        if (prompt) formData.append('prompt', prompt);

        fetch(`${API_BASE}/process`, { method: 'POST', body: formData })
            .then(res => res.json().then(data => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) throw new Error(data.error || 'Upload failed');
                currentSessionId = data.session_id;
                ub.innerText = 'Processing\u2026';
                pollStatus(currentSessionId);
            })
            .catch(err => {
                alert(`Error: ${err.message}`);
                ub.disabled = false;
                ub.innerText = 'Generate Scene';
                pb.style.display = 'none';
                pf.style.width = '0%';
                pt.innerText = '';
            });
    });
    ms.addEventListener('change', () => {
        pr.style.display = PROMPT_MODELS.includes(ms.value) ? 'flex' : 'none';
    });
    lb.addEventListener('click', loadTestScene);
}

// ── Orbit Frame Viewer ──────────────────────────────────────────────
async function loadOrbitImages(viewUrls) {
    orbitState.images = [];
    orbitState.loaded = false;

    const promises = viewUrls.map(url => {
        return new Promise((resolve, reject) => {
            const img = new window.Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = `${API_BASE}${url}`;
        });
    });

    orbitState.images = await Promise.all(promises);
    orbitState.loaded = true;
    orbitState.currentIndex = 0;

    sceneLoaded = true;
    blocker.style.display = 'none';
    modeBar.style.display = 'flex';
    switchMode('orbit');
    drawOrbitFrame();
}

function drawOrbitFrame() {
    if (!orbitState.loaded || orbitState.images.length === 0) return;

    const img = orbitState.images[orbitState.currentIndex];
    const canvas = orbitCanvas;
    const ctx = orbitState.ctx;

    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    canvas.style.width = `${window.innerWidth}px`;
    canvas.style.height = `${window.innerHeight}px`;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Fit image to canvas maintaining aspect ratio
    const imgAspect = img.width / img.height;
    const canvasAspect = canvas.width / canvas.height;
    let drawW, drawH, drawX, drawY;

    if (imgAspect > canvasAspect) {
        drawW = canvas.width;
        drawH = canvas.width / imgAspect;
        drawX = 0;
        drawY = (canvas.height - drawH) / 2;
    } else {
        drawH = canvas.height;
        drawW = canvas.height * imgAspect;
        drawX = (canvas.width - drawW) / 2;
        drawY = 0;
    }

    ctx.drawImage(img, drawX, drawY, drawW, drawH);

    orbitFrameCounter.innerText = `Frame ${orbitState.currentIndex + 1} / ${orbitState.images.length}`;
}

function onOrbitMouseDown(e) {
    if (!orbitState.loaded) return;
    orbitState.isDragging = true;
    orbitState.startX = e.clientX;
    orbitState.startIndex = orbitState.currentIndex;
    orbitViewer.style.cursor = 'grabbing';
}

function onOrbitMouseMove(e) {
    if (!orbitState.isDragging || !orbitState.loaded) return;
    const dx = e.clientX - orbitState.startX;
    const sensitivity = window.innerWidth / orbitState.images.length;
    const indexDelta = Math.round(dx / sensitivity);
    let newIndex = orbitState.startIndex + indexDelta;
    // Wrap around for continuous orbiting
    const len = orbitState.images.length;
    newIndex = ((newIndex % len) + len) % len;
    if (newIndex !== orbitState.currentIndex) {
        orbitState.currentIndex = newIndex;
        drawOrbitFrame();
    }
}

function onOrbitMouseUp() {
    orbitState.isDragging = false;
    orbitViewer.style.cursor = 'grab';
}

function onOrbitTouchStart(e) {
    if (!orbitState.loaded || e.touches.length !== 1) return;
    e.preventDefault();
    orbitState.isDragging = true;
    orbitState.startX = e.touches[0].clientX;
    orbitState.startIndex = orbitState.currentIndex;
}

function onOrbitTouchMove(e) {
    if (!orbitState.isDragging || !orbitState.loaded) return;
    e.preventDefault();
    const dx = e.touches[0].clientX - orbitState.startX;
    const sensitivity = window.innerWidth / orbitState.images.length;
    const indexDelta = Math.round(dx / sensitivity);
    let newIndex = orbitState.startIndex + indexDelta;
    const len = orbitState.images.length;
    newIndex = ((newIndex % len) + len) % len;
    if (newIndex !== orbitState.currentIndex) {
        orbitState.currentIndex = newIndex;
        drawOrbitFrame();
    }
}

function onOrbitTouchEnd() {
    orbitState.isDragging = false;
}

// ── Mode Switching ──────────────────────────────────────────────────
function switchMode(mode) {
    currentMode = mode;

    if (mode === 'orbit') {
        modeOrbitBtn.classList.add('active');
        mode3dBtn.classList.remove('active');
        orbitViewer.style.display = 'block';
        orbitFrameCounter.style.display = 'block';
        renderer.domElement.style.display = 'none';
        if (crosshair) crosshair.style.display = 'none';
        if (controls.isLocked) controls.unlock();
        info.innerText = 'Drag left/right to orbit around the scene';
        info.style.display = 'block';
        drawOrbitFrame();
    } else {
        modeOrbitBtn.classList.remove('active');
        mode3dBtn.classList.add('active');
        orbitViewer.style.display = 'none';
        orbitFrameCounter.style.display = 'none';
        renderer.domElement.style.display = 'block';
        info.innerText = 'Click to enter 3D view. WASD to move, Mouse to look, E/Q up/down, Shift sprint.';
        info.style.display = 'block';
    }
}

// ── FPS Controls (unchanged) ────────────────────────────────────────
function setupControls() {
    controls.addEventListener('lock', () => {
        if (sceneLoaded && currentMode === '3d') {
            blocker.style.display = 'none';
            if (crosshair) crosshair.style.display = 'block';
        }
    });

    controls.addEventListener('unlock', () => {
        if (sceneLoaded && currentMode === '3d') {
            blocker.style.display = 'flex';
            blocker.querySelector('.upload-container').innerHTML = `
                <h1>Paused</h1>
                <p>Click to resume</p>
                <p class="controls-hint">WASD — Move &nbsp;|&nbsp; Mouse — Look &nbsp;|&nbsp; E/Q — Up/Down &nbsp;|&nbsp; Shift — Sprint</p>
            `;
            if (crosshair) crosshair.style.display = 'none';
        }
    });

    blocker.addEventListener('click', (e) => {
        if (sceneLoaded && currentMode === '3d' && !e.target.closest('button') && !e.target.closest('select') && !e.target.closest('input')) {
            controls.lock();
        }
    });
}

function onKeyDown(event) {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
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
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
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
    if (currentMode === 'orbit' && orbitState.loaded) {
        drawOrbitFrame();
    }
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

// ── GLB Loading (shared by test scene + pipeline) ───────────────────
function loadGLB(glbUrl) {
    const loader = new GLTFLoader();

    loader.load(
        glbUrl,
        function (gltf) {
            const model = gltf.scene;
            console.log("Loaded GLTF Model:", model);

            model.traverse((child) => {
                if (child.isMesh) {
                    child.material = new THREE.MeshBasicMaterial({
                        vertexColors: true,
                        side: THREE.DoubleSide
                    });
                }
            });

            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());

            model.position.x += (model.position.x - center.x);
            model.position.y += (model.position.y - center.y);
            model.position.z += (model.position.z - center.z);

            scene.add(model);
            camera.position.set(0, 0, 2);

            sceneLoaded = true;

            // If we don't have orbit frames, go straight to 3D mode
            if (!orbitState.loaded) {
                blocker.style.display = 'none';
                modeBar.style.display = 'flex';
                switchMode('3d');
            }
        },
        function (xhr) {
            if (xhr.total) {
                const percent = Math.round(xhr.loaded / xhr.total * 100);
                progressText.innerText = `Loading 3D model: ${percent}%`;
            }
        },
        function (error) {
            console.error('Error loading GLB:', error);
        }
    );
}

// ── Test Scene Loading ──────────────────────────────────────────────
function loadTestScene() {
    loadBtn.disabled = true;
    loadBtn.innerText = 'Loading…';
    info.style.display = 'block';
    progressBar.style.display = 'block';

    // Try to load orbit views from test_scene
    fetch(`${API_BASE}/views/test_scene`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
            if (data && data.views && data.views.length > 0) {
                return loadOrbitImages(data.views);
            }
        })
        .catch(() => {});

    // Load GLB
    loadGLB(`${API_BASE}/uploads/test_scene/scene.glb`);
}
