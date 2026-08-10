@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: PHASE 2 AUTOMATION & DEPENDENCY BOOTSTRAPPER
:: ============================================================================
:: EXPERIMENTAL FORWARD-ONLY REFERENCE. This target is not the trainer backend;
:: backward.cu and the PyTorch binding remain unimplemented.
:: Double-click this script from anywhere on the NVIDIA GPU PC.
:: ============================================================================

:: Resolve the repository root from renderer-cuda\run_phase2.bat itself.
for %%I in ("%~dp0..") do set "PROJECT_DIR=%%~fI"

echo [1/6] Verifying Project Directory...
if not exist "%PROJECT_DIR%" (
    echo [ERROR] Project directory "%PROJECT_DIR%" does not exist.
    echo The repository-relative path could not be resolved.
    pause
    exit /b 1
)
cd /d "%PROJECT_DIR%\renderer-cuda"

echo [2/6] Checking for PLY input...
if not exist "..\scenes\iceland_splat.ply" (
    echo [WARNING] "scenes\iceland_splat.ply" was not found in the root of the project.
    echo The renderer will fall back to generating mock Gaussian data for verification.
    echo If you want to render a real scene, place your PLY file at:
    echo   %PROJECT_DIR%\scenes\iceland_splat.ply
)

echo [3/6] Setting Up Environment / Compiler Detection...

:: 1. Check for CUDA Compiler (NVCC)
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo NVCC not in PATH. Searching standard installation directories...
    
    :: Search for CUDA Toolkit versions in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\
    set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if exist "!CUDA_ROOT!" (
        for /f "delims=" %%I in ('dir /b /ad "!CUDA_ROOT!\v*" 2^>nul') do (
            set "CUDA_PATH=!CUDA_ROOT!\%%I"
        )
    )
    
    if defined CUDA_PATH (
        echo Found CUDA Toolkit at: !CUDA_PATH!
        set "PATH=!CUDA_PATH!\bin;!PATH!"
        set "CUDA_BIN=!CUDA_PATH!\bin"
    ) else (
        echo [ERROR] CUDA Toolkit (NVCC) could not be found.
        echo Please ensure CUDA is installed. Download it here:
        echo   https://developer.nvidia.com/cuda-downloads
        pause
        exit /b 1
    )
) else (
    for /f "delims=" %%I in ('where nvcc') do set "CUDA_BIN=%%~dpI"
    echo CUDA Compiler (NVCC) is active.
)

:: 2. Detect C++ host compiler (cl.exe for MSVC or g++ for MinGW)
set "COMPILER_FOUND=0"
where cl >nul 2>nul
if %errorlevel% equ 0 (
    echo MSVC compiler (cl) is already active in this shell.
    set "COMPILER_FOUND=1"
) else (
    echo Searching for Visual Studio C++ Compiler environment setup scripts...
    :: Common paths for vcvars64.bat in VS 2022 / 2019 / Build Tools
    for %%P in (
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if exist %%P (
            echo Loading C++ build tools from: %%P
            call %%P
            set "COMPILER_FOUND=1"
            goto compiler_checked
        )
    )
)

:compiler_checked
if "%COMPILER_FOUND%"=="0" (
    :: Check if g++ is available as a fallback host compiler
    where g++ >nul 2>nul
    if %errorlevel% equ 0 (
        echo Visual Studio tools not found, but GCC/G++ is available. Using MinGW.
        set "COMPILER_FOUND=2"
    ) else (
        echo [WARNING] No host C++ compiler (MSVC 'cl' or MinGW 'g++') was found.
        echo NVCC requires a host compiler. Compilation may fail unless one is installed.
    )
)

echo [4/6] Creating Build Directories...
if not exist "build" mkdir build

echo [5/6] Building Phase 2 CUDA Renderer...

set "BUILD_SUCCESS=0"

:: Attempt CMake build first, if cmake is available
where cmake >nul 2>nul
if %errorlevel% equ 0 (
    echo CMake detected. Running CMake project build...
    cmake -B build -S .
    if !errorlevel! equ 0 (
        cmake --build build --config Release
        if !errorlevel! equ 0 (
            set "BUILD_SUCCESS=1"
            set "EXEC_PATH=build\Release\renderer.exe"
            if not exist "!EXEC_PATH!" set "EXEC_PATH=build\renderer.exe"
            goto run_stage
        )
    )
    echo [WARNING] CMake build failed. Trying fallback manual compilation...
) else (
    echo CMake not found in PATH. Trying fallback manual compilation...
)

:: Fallback manual compilation command using NVCC directly
echo Compiling directly with NVCC...
if "%COMPILER_FOUND%"=="2" (
    :: MinGW g++ fallback
    nvcc -O3 -std=c++17 -ccbin g++ src\main.cpp src\forward.cu src\backward.cu -o build\renderer.exe
) else (
    :: Default/MSVC fallback
    nvcc -O3 -std=c++17 src\main.cpp src\forward.cu src\backward.cu -o build\renderer.exe
)

if %errorlevel% equ 0 (
    set "BUILD_SUCCESS=1"
    set "EXEC_PATH=build\renderer.exe"
) else (
    echo [ERROR] Direct NVCC compilation failed.
    echo Please ensure that both CUDA Toolkit and a C++ compiler are installed.
    pause
    exit /b 1
)

:run_stage
if "%BUILD_SUCCESS%"=="1" (
    echo [6/6] Running Phase 2 Renderer...
    if exist "!EXEC_PATH!" (
        cd build
        :: Run the executable (we run from build/ directory so relative path "../../scenes/" works)
        "..\"!EXEC_PATH!
        if !errorlevel! equ 0 (
            echo [SUCCESS] Renderer executed successfully.
            :: Check if output is saved
            if exist "rendered_output.ppm" (
                echo Rendered image saved to:
                echo   %PROJECT_DIR%\renderer-cuda\build\rendered_output.ppm
            ) else if exist "..\rendered_output.ppm" (
                echo Rendered image saved to:
                echo   %PROJECT_DIR%\renderer-cuda\rendered_output.ppm
            )
        ) else (
            echo [ERROR] Renderer execution failed.
        )
    ) else (
        echo [ERROR] Compiled binary not found at: !EXEC_PATH!
    )
)

pause
