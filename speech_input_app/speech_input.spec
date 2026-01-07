# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec-Datei für Speech Input App
Erstellt eine standalone ausführbare Datei
"""
import os

block_cipher = None

a = Analysis(
    ['speech_input.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config.json', '.'),
    ] + ([('icons', 'icons')] if os.path.exists('icons') else []),
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'sounddevice',
        'numpy',
        'scipy',
        'scipy.io.wavfile',
        'keyboard',
        'win32gui',
        'win32con',
        'win32clipboard',
        'requests',
        'sqlite3',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='speech_input',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Kein Konsolenfenster (GUI-App)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Icon kann später hinzugefügt werden, falls vorhanden
)

