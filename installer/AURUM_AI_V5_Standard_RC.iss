; AURUM-AI V5 - Windows installer (research / historical simulation)
; Compile from the project with tools/windows_desktop/build_installer_phase23.py.
; NEVER distribute private SQLite databases, account data or SMTP secrets.

#define ApplicationName "AURUM-AI V5 Standard"
#define ApplicationVersion "0.9.0"
#define ApplicationExe "AURUM-AI V5.exe"

[Setup]
AppId={{2046CD8D-4650-4CD7-88C4-31C7844FE805}
AppName={#ApplicationName}
AppVersion={#ApplicationVersion}
AppPublisher=AURUM-AI Academic Research
AppVerName={#ApplicationName} (Desktop Preview {#ApplicationVersion})
DefaultDirName={localappdata}\Programs\AURUM-AI V5 Standard
DefaultGroupName=AURUM-AI V5 Standard
PrivilegesRequired=lowest
OutputDir=..\dist_installer_academic_rc
OutputBaseFilename=AURUM-AI-V5-Standard-0.9.0-rc1
SetupIconFile=..\assets\branding\aurum_ai.ico
UninstallDisplayIcon={app}\{#ApplicationExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
ArchitecturesAllowed=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Creer un raccourci sur le Bureau"; GroupDescription: "Raccourcis :"; Flags: unchecked

[Files]
; Exclusions guard against bundling passwords/accounts and private configuration.
; SQLite AUTH is stored separately in %LOCALAPPDATA%\AURUM-AI\auth\users.sqlite3
; The installer must NOT copy it and the uninstaller must NOT remove it.
Source: "..\dist\AURUM-AI V5\*"; DestDir: "{app}"; Excludes: "*.sqlite3,*.sqlite,*.db,.env,.env.*,secrets.toml,*.p12,*.pfx,*_before_*.py,*_backup.py,*_old.py,*_bak.py,*_copy.py,*_archive.py,*-backup.py,*-before-*.py,backup_*.py,before_*.py,old_*.py,copy_*.py,archive_*.py"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\AURUM-AI V5 Standard"; Filename: "{app}\{#ApplicationExe}"; WorkingDir: "{app}"; IconFilename: "{app}\{#ApplicationExe}"
Name: "{autodesktop}\AURUM-AI V5 Standard"; Filename: "{app}\{#ApplicationExe}"; WorkingDir: "{app}"; IconFilename: "{app}\{#ApplicationExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ApplicationExe}"; Description: "Lancer AURUM-AI V5"; Flags: nowait postinstall skipifsilent

; User databases in %LOCALAPPDATA%\AURUM-AI are intentionally retained by the uninstaller.

