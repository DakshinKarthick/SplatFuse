:: ================================================================
::  FAST BATCH PHOTOGRAMMETRY TRACKER  (fps-decimated variant)
::  Based on polyfjord's batch_reconstruct.bat
::  https://gist.github.com/polyfjord/4ed7e8988bdb9674145f1c270440200d
:: ================================================================
::  WHY THIS EXISTS
::    polyfjord's original extracts EVERY frame, which makes COLMAP's
::    (CPU) matching crawl on long clips. Structure-from-motion needs
::    far fewer frames than 30/60 fps. This variant decimates to FPS
::    frames/sec (see the knob below) so a run takes minutes, not hours.
::
::    It also bakes in two fixes needed on the dev machine (AMD RX 7600
::    + a Parsec virtual display): QT_PLUGIN_PATH so COLMAP finds its Qt
::    plugin, and CPU matching so the OpenGL SIFT matcher can't grab the
::    virtual adapter and die. See phase0-capture/README.md.
::
::  FOLDER LAYOUT (identical to the original, folders sit side-by-side):
::    01 COLMAP  - COLMAP release (colmap.exe under \bin, plus \plugins)
::    02 VIDEOS  - input videos (.mp4, .mov, ...)
::    03 FFMPEG  - static FFmpeg build (ffmpeg.exe or bin\ffmpeg.exe)
::    04 SCENES  - output, one sub-folder per video
::    05 SCRIPTS - this file
:: ================================================================
@echo off

:: ================================================================
::  KNOB -- frames per second to extract. Lower = faster + fewer
::  cameras; higher = smoother fly-through but slower matching.
::  6 is a good default for a tracked camera. Set higher (12-15) if
::  you want a denser per-frame camera path.
:: ================================================================
set "FPS=6"

:: ---------- Resolve top-level folder (one up from this .bat) -----
pushd "%~dp0\.." >nul
set "TOP=%cd%"
popd >nul

:: ---------- Key paths -------------------------------------------
set "COLMAP_DIR=%TOP%\01 COLMAP"
set "VIDEOS_DIR=%TOP%\02 VIDEOS"
set "FFMPEG_DIR=%TOP%\03 FFMPEG"
set "SCENES_DIR=%TOP%\04 SCENES"

:: ---------- Locate ffmpeg.exe -----------------------------------
if exist "%FFMPEG_DIR%\ffmpeg.exe" (
    set "FFMPEG=%FFMPEG_DIR%\ffmpeg.exe"
) else if exist "%FFMPEG_DIR%\bin\ffmpeg.exe" (
    set "FFMPEG=%FFMPEG_DIR%\bin\ffmpeg.exe"
) else (
    echo [ERROR] ffmpeg.exe not found inside "%FFMPEG_DIR%".
    pause & goto :eof
)

:: ---------- Locate colmap.exe (skip the .bat) --------------------
if exist "%COLMAP_DIR%\colmap.exe" (
    set "COLMAP=%COLMAP_DIR%\colmap.exe"
) else if exist "%COLMAP_DIR%\bin\colmap.exe" (
    set "COLMAP=%COLMAP_DIR%\bin\colmap.exe"
) else (
    echo [ERROR] colmap.exe not found inside "%COLMAP_DIR%".
    pause & goto :eof
)

:: ---------- Put COLMAP's dll folder(s) on PATH -------------------
set "PATH=%COLMAP_DIR%;%COLMAP_DIR%\bin;%PATH%"

:: ---------- Point Qt at COLMAP's plugin folder ------------------
:: COLMAP 3.12.x ships qwindows.dll under \plugins; without this the
:: feature_extractor / sequential_matcher fail to start.
if exist "%COLMAP_DIR%\plugins" (
    set "QT_PLUGIN_PATH=%COLMAP_DIR%\plugins;%QT_PLUGIN_PATH%"
) else if exist "%COLMAP_DIR%\bin\plugins" (
    set "QT_PLUGIN_PATH=%COLMAP_DIR%\bin\plugins;%QT_PLUGIN_PATH%"
)

:: ---------- Ensure required folders exist ------------------------
if not exist "%VIDEOS_DIR%" (
    echo [ERROR] Input folder "%VIDEOS_DIR%" missing.
    pause & goto :eof
)
if not exist "%SCENES_DIR%" mkdir "%SCENES_DIR%"

:: ---------- Count videos for progress bar ------------------------
for /f %%C in ('dir /b /a-d "%VIDEOS_DIR%\*" ^| find /c /v ""') do set "TOTAL=%%C"
if "%TOTAL%"=="0" (
    echo [INFO] No video files found in "%VIDEOS_DIR%".
    pause & goto :eof
)

echo ==============================================================
echo  FAST tracker - %TOTAL% video(s), extracting at %FPS% fps
echo ==============================================================

setlocal EnableDelayedExpansion
set /a IDX=0

for %%V in ("%VIDEOS_DIR%\*.*") do (
    if exist "%%~fV" (
        set /a IDX+=1
        call :PROCESS_VIDEO "%%~fV" "!IDX!" "%TOTAL%"
    )
)

echo --------------------------------------------------------------
echo  All jobs finished - results are in "%SCENES_DIR%".
echo --------------------------------------------------------------
pause
goto :eof


:PROCESS_VIDEO
:: ----------------------------------------------------------------
::  %1 = full path to video   %2 = current index   %3 = total
:: ----------------------------------------------------------------
setlocal
set "VIDEO=%~1"
set "NUM=%~2"
set "TOT=%~3"

for %%I in ("%VIDEO%") do (
    set "BASE=%%~nI"
    set "EXT=%%~xI"
)

echo.
echo [!NUM!/!TOT!] === Processing "!BASE!!EXT!" ===

:: -------- Directory layout for this scene -----------------------
set "SCENE=%SCENES_DIR%\!BASE!"
set "IMG_DIR=!SCENE!\images"
set "SPARSE_DIR=!SCENE!\sparse"

:: -------- Skip if already reconstructed -------------------------
if exist "!SCENE!" (
    echo        Skipping "!BASE!" - already reconstructed.
    goto :END
)

:: Clean slate ----------------------------------------------------
mkdir "!IMG_DIR!"   >nul
mkdir "!SPARSE_DIR!" >nul

:: -------- 1) Extract frames at FPS ------------------------------
echo        [1/4] Extracting frames at %FPS% fps ...
"%FFMPEG%" -loglevel error -stats -i "!VIDEO!" -r %FPS% -qscale:v 2 ^
    "!IMG_DIR!\frame_%%06d.jpg"
if errorlevel 1 (
    echo        [X] FFmpeg failed - skipping "!BASE!".
    goto :END
)
dir /b "!IMG_DIR!\*.jpg" >nul 2>&1 || (
    echo        [X] No frames extracted - skipping "!BASE!".
    goto :END
)

:: -------- 2) Feature extraction (GPU) ---------------------------
echo        [2/4] COLMAP feature_extractor (GPU) ...
"%COLMAP%" feature_extractor ^
    --database_path "!SCENE!\database.db" ^
    --image_path    "!IMG_DIR!" ^
    --ImageReader.single_camera 1 ^
    --SiftExtraction.use_gpu 1 ^
    --SiftExtraction.max_image_size 4096
if errorlevel 1 (
    echo        [X] feature_extractor failed - skipping "!BASE!".
    goto :END
)

:: -------- 3) Sequential matching (CPU) --------------------------
echo        [3/4] COLMAP sequential_matcher (CPU) ...
"%COLMAP%" sequential_matcher ^
    --database_path "!SCENE!\database.db" ^
    --SiftMatching.use_gpu 0 ^
    --SequentialMatching.overlap 15
if errorlevel 1 (
    echo        [X] sequential_matcher failed - skipping "!BASE!".
    goto :END
)

:: -------- 4) Sparse reconstruction ------------------------------
echo        [4/4] COLMAP mapper ...
"%COLMAP%" mapper ^
    --database_path "!SCENE!\database.db" ^
    --image_path    "!IMG_DIR!" ^
    --output_path   "!SPARSE_DIR!" ^
    --Mapper.num_threads %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo        [X] mapper failed - skipping "!BASE!".
    goto :END
)

:: -------- Identify the biggest sub-model ------------------------
:: COLMAP writes one numbered folder per disconnected chunk (sparse\0,
:: sparse\1, ...). The one with the largest points3D.bin is the best.
set "BEST="
set "BEST_SIZE=0"
for /d %%M in ("!SPARSE_DIR!\*") do (
    if exist "%%M\points3D.bin" (
        for %%S in ("%%M\points3D.bin") do (
            if %%~zS GTR !BEST_SIZE! (
                set "BEST_SIZE=%%~zS"
                set "BEST=%%M"
            )
        )
    )
)

:: -------- Export best model as TXT into sparse\ root ------------
:: The Blender SBCV importer's *workspace* check looks for the model
:: files (cameras/images/points3D) DIRECTLY inside sparse\ - it does
:: NOT descend into sparse\0. Writing the best model's .txt there lets
:: you just import the SCENE folder and have it auto-load images\.
if defined BEST (
    "%COLMAP%" model_converter ^
        --input_path  "!BEST!" ^
        --output_path "!SPARSE_DIR!" ^
        --output_type TXT >nul
    echo        [OK] Finished "!BASE!"  (!NUM!/!TOT!)
    echo        ==^> In Blender, File ^> Import ^> COLMAP model/workspace and pick:
    echo            "!SCENE!"
    echo            (best of several sub-models exported; raw chunks remain in sparse\0,1,...)
) else (
    echo        [X] mapper produced no usable model for "!BASE!".
)

:END
endlocal & goto :eof
