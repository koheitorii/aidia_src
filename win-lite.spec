# -*- mode: python ; coding: utf-8 -*-

import sys
sys.setrecursionlimit(5000)  # required on Windows

block_cipher = None

a = Analysis(
    ['aidia/__main__.py'],
    pathex=['aidia'],
    binaries=[],
    datas=[
        ('aidia/config/default_config.yaml', 'aidia/config'),
        ('aidia/icons/*', 'aidia/icons'),
        ('aidia/translate/ja_JP.qm', 'aidia/translate'),
    ],
    hiddenimports=['pydicom.encoders.gdcm', 'pydicom.encoders.pylibjpeg'],
    excludes=[
        'vs2015_runtime',
        'pyinstaller',
        'pyinstaller-versionfile',
        'pip',
        'setuptools',
        'vc',
        'wheel',
        'powershell_shortcut'
        ],
    hookspath=[],
    runtime_hooks=[],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Aidia-lite',
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon='aidia/icons/icon-lite.ico',
    codesign_identify=None,
    entitlements_file=None,
    version='version.txt',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Aidia-lite',
)
app = BUNDLE(
    exe,
    name='Aidia-lite.app',
    icon='aidia/icons/icon-lite.ico',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)