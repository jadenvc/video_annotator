# PyInstaller spec for EdgeTAM Tracker
# Build with: bash build_app.sh
#
# Hydra note: sam2/__init__.py calls initialize_config_module("sam2"), so hydra
# resolves configs from inside the sam2 package. We bundle sam2/configs/ and the
# top-level sam2/*.yaml files so that path stays intact in the frozen app.
#
# Checkpoint note: run_tracker.py is patched to use os.path.dirname(sys.executable)
# when sys.frozen is set, which points to Contents/MacOS/ inside the .app where
# checkpoints/edgetam.pt is placed by COLLECT.

from PyInstaller.utils.hooks import collect_all

block_cipher = None

torch_datas,      torch_bins,      torch_hidden      = collect_all('torch')
torchvision_datas, torchvision_bins, torchvision_hidden = collect_all('torchvision')
timm_datas,       timm_bins,       timm_hidden       = collect_all('timm')

a = Analysis(
    ['run_tracker.py'],
    pathex=['.'],
    binaries=torch_bins + torchvision_bins + timm_bins,
    datas=(
        torch_datas
        + torchvision_datas
        + timm_datas
        + [
            # hydra config files that sam2 registers via initialize_config_module
            ('sam2/configs',   'sam2/configs'),
            ('sam2/*.yaml',    'sam2'),
            # model checkpoint — ends up at Contents/MacOS/checkpoints/edgetam.pt
            ('checkpoints/edgetam.pt', 'checkpoints'),
        ]
    ),
    hiddenimports=(
        torch_hidden
        + torchvision_hidden
        + timm_hidden
        + [
            # sam2 package (editable install — must be explicit)
            'sam2',
            'sam2.build_sam',
            'sam2.sam2_video_predictor',
            'sam2.sam2_image_predictor',
            'sam2.automatic_mask_generator',
            'sam2.modeling',
            'sam2.modeling.backbones',
            'sam2.modeling.backbones.hieradet',
            'sam2.modeling.backbones.image_encoder',
            'sam2.modeling.backbones.timm',
            'sam2.modeling.backbones.utils',
            'sam2.modeling.memory_attention',
            'sam2.modeling.memory_encoder',
            'sam2.modeling.perceiver',
            'sam2.modeling.position_encoding',
            'sam2.modeling.sam',
            'sam2.modeling.sam.mask_decoder',
            'sam2.modeling.sam.prompt_encoder',
            'sam2.modeling.sam.transformer',
            'sam2.modeling.sam2_base',
            'sam2.modeling.sam2_utils',
            'sam2.utils',
            'sam2.utils.amg',
            'sam2.utils.misc',
            'sam2.utils.transforms',
            # hydra / omegaconf
            'hydra',
            'hydra._internal',
            'hydra._internal.config_loader_impl',
            'hydra._internal.utils',
            'hydra.core',
            'hydra.core.global_hydra',
            'hydra.core.plugins',
            'hydra.core.utils',
            'hydra.utils',
            'omegaconf',
            # Qt / cv2 / matplotlib
            'PyQt6',
            'PyQt6.QtCore',
            'PyQt6.QtGui',
            'PyQt6.QtWidgets',
            'cv2',
            'matplotlib',
            'matplotlib.backends.backend_agg',
        ]
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['decord', 'eva_decord'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='EdgeTAM Tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX breaks torch dylibs on Mac
    console=False,      # no terminal window
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='EdgeTAM Tracker',
)

app = BUNDLE(
    coll,
    name='EdgeTAM Tracker.app',
    icon=None,
    bundle_identifier='com.ctag.edgetam-tracker',
    info_plist={
        'CFBundleDisplayName': 'EdgeTAM Tracker',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
    },
)
