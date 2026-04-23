$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\Kola_Yeswanth\OneDrive\Desktop\fyp'
$MobileRoot = Join-Path $ProjectRoot 'mobile-android'
$SdkRoot = Join-Path $env:LOCALAPPDATA 'Android\Sdk'
$CmdlineZip = Join-Path $env:TEMP 'commandlinetools-win.zip'
$CmdlineToolsRoot = Join-Path $SdkRoot 'cmdline-tools'
$CmdlineLatest = Join-Path $CmdlineToolsRoot 'latest'
$CmdlineDownloadUrl = 'https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip'

Write-Host '== EatWise Android setup starting =='

function Add-UserPath([string]$PathEntry) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    $parts = $userPath.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($parts -notcontains $PathEntry) {
        $newPath = ($parts + $PathEntry) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Write-Host "Added to USER PATH: $PathEntry"
    }
}

function Set-UserEnv([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, 'User')
    Set-Item -Path "Env:$Name" -Value $Value
    Write-Host "Set $Name=$Value"
}

function Resolve-JavaHome {
    $candidates = @(
        'C:\Program Files\Android\Android Studio\jbr',
        'C:\Program Files\Microsoft\jdk-17',
        'C:\Program Files\OpenJDK\jdk-17',
        'C:\Program Files\Eclipse Adoptium\jdk-17',
        'C:\Program Files\Java\jdk-17'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'bin\java.exe')) {
            return $candidate
        }
    }

    $javaCommand = Get-Command java -ErrorAction SilentlyContinue
    if ($javaCommand) {
        $binPath = Split-Path -Parent $javaCommand.Source
        return Split-Path -Parent $binPath
    }

    throw 'Unable to locate a Java installation.'
}

Write-Host 'Installing required packages via winget (OpenJDK, Android Studio, PlatformTools)...'
winget install -e --id Microsoft.OpenJDK.17 --accept-package-agreements --accept-source-agreements --disable-interactivity
winget install -e --id Google.AndroidStudio --accept-package-agreements --accept-source-agreements --disable-interactivity
winget install -e --id Google.PlatformTools --accept-package-agreements --accept-source-agreements --disable-interactivity

$JavaHome = Resolve-JavaHome
Set-UserEnv 'JAVA_HOME' $JavaHome
Add-UserPath "$JavaHome\bin"

if (-not (Test-Path 'C:\Gradle\gradle-8.10.2\bin\gradle.bat')) {
    Write-Host 'Installing Gradle 8.10.2 manually...'
    $gradleZip = Join-Path $env:TEMP 'gradle-8.10.2-bin.zip'
    Invoke-WebRequest -Uri 'https://services.gradle.org/distributions/gradle-8.10.2-bin.zip' -OutFile $gradleZip
    New-Item -ItemType Directory -Force -Path 'C:\Gradle' | Out-Null
    Expand-Archive -Path $gradleZip -DestinationPath 'C:\Gradle' -Force
}
Add-UserPath 'C:\Gradle\gradle-8.10.2\bin'

New-Item -ItemType Directory -Force -Path $CmdlineToolsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $SdkRoot | Out-Null

Write-Host 'Downloading Android command line tools...'
if (Test-Path $CmdlineZip) { Remove-Item $CmdlineZip -Force }
Start-BitsTransfer -Source $CmdlineDownloadUrl -Destination $CmdlineZip -DisplayName 'eatwise-android-cmdline-tools'

if ((Get-Item $CmdlineZip).Length -lt 100MB) {
    throw 'Android command line tools download seems incomplete. Please re-run script with stable internet.'
}

if (Test-Path $CmdlineLatest) {
    Remove-Item $CmdlineLatest -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $CmdlineLatest | Out-Null

$tmpExtract = Join-Path $env:TEMP 'eatwise-cmdline-extract'
if (Test-Path $tmpExtract) { Remove-Item $tmpExtract -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tmpExtract | Out-Null
Expand-Archive -Path $CmdlineZip -DestinationPath $tmpExtract -Force

$nestedRoot = Join-Path $tmpExtract 'cmdline-tools'
if (Test-Path $nestedRoot) {
    Get-ChildItem -Path $nestedRoot -Force | ForEach-Object {
        Move-Item -Path $_.FullName -Destination $CmdlineLatest -Force
    }
} else {
    Get-ChildItem -Path $tmpExtract -Force | ForEach-Object {
        Move-Item -Path $_.FullName -Destination $CmdlineLatest -Force
    }
}

[Environment]::SetEnvironmentVariable('ANDROID_HOME', $SdkRoot, 'User')
[Environment]::SetEnvironmentVariable('ANDROID_SDK_ROOT', $SdkRoot, 'User')
Add-UserPath "$SdkRoot\platform-tools"
Add-UserPath "$SdkRoot\cmdline-tools\latest\bin"

$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
$env:JAVA_HOME = $JavaHome

Write-Host 'Verifying sdkmanager...'
& "$SdkRoot\cmdline-tools\latest\bin\sdkmanager.bat" --version

Write-Host 'Accepting Android SDK licenses...'
cmd /c "for /L %i in (1,1,200) do @echo y" | & "$SdkRoot\cmdline-tools\latest\bin\sdkmanager.bat" --licenses

Write-Host 'Installing Android SDK packages...'
& "$SdkRoot\cmdline-tools\latest\bin\sdkmanager.bat" "platform-tools" "platforms;android-34" "build-tools;34.0.0"

Write-Host 'Writing local.properties...'
$localProps = Join-Path $MobileRoot 'local.properties'
"sdk.dir=$($SdkRoot -replace '\\','\\\\')" | Out-File -FilePath $localProps -Encoding ascii -Force

Write-Host 'Generating Gradle wrapper and building debug APK...'
Push-Location $MobileRoot
try {
    gradle wrapper
    .\gradlew.bat assembleDebug
} finally {
    Pop-Location
}

$apkPath = Join-Path $MobileRoot 'app\build\outputs\apk\debug\app-debug.apk'
if (Test-Path $apkPath) {
    Write-Host "APK built successfully: $apkPath"
} else {
    throw 'APK build finished without app-debug.apk output. Check Gradle logs above.'
}

Write-Host '== Setup complete =='
