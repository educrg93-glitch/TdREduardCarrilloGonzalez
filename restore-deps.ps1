# restore-deps.ps1
# Crea un entorn virtual a .venv (si s'escau) i instal·la les dependències
# utilitzant l'intèrpret de Python del venv per evitar dependre de 'pip' a la ruta.

$venvDir = ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Output "Creant un entorn virtual a $venvDir..."
    python -m venv $venvDir
}

$pythonExe = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Executable de Python no trobat a $pythonExe. Assegureu-vos que Python està a la ruta i torneu-ho a provar."
    exit 1
}

Write-Output "Actualitzant pip al venv..."
& $pythonExe -m pip install --upgrade pip

Write-Output "Instal·lant requisits des de requeriments.txt..."
& $pythonExe -m pip install -r requeriments.txt

Write-Output "Fet. Per activar el venv a PowerShell, executeu:`n.\\.venv\\Scripts\\Activate.ps1"