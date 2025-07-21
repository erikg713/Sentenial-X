from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files

hidden_imports = collect_submodules('sentenial_core') + collect_submodules('plugins')

block_cipher = None

a = Analysis(
    ['gui/hub.py'],
    pathex=['.'],
    binaries=[],
    datas=collect_data_files('sentenial_core') + collect_data_files('plugins'),
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CentennialX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # change to True if you want console output
    icon='assets/icon.ico'  # optional: path to your app icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CentennialX'
)