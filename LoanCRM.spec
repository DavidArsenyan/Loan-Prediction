# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['E:\\Davit\\UFAR\\4th course\\idbank\\my_project_V3\\scripts\\crm_app3.py'],
    pathex=[],
    binaries=[],
    datas=[('E:\\Davit\\UFAR\\4th course\\idbank\\my_project_V3\\data', 'data')],
    hiddenimports=['sklearn.utils._cython_blas', 'sklearn.neighbors._typedefs', 'sklearn.tree._classes', 'lightgbm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LoanCRM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
