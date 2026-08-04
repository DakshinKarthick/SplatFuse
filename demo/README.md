# SplatFuse viewer demo

The custom Three.js viewer is self-contained except for trained `.ply` scene files, which are intentionally ignored by Git. The portable data zip contains `mypic1.ply` (the default) and `mypic2.ply` under their exact restore path: `GaussianSplat/viewer/public/scenes/`.

After restoring the zip, run `powershell -ExecutionPolicy Bypass -File .\demo\run-viewer.ps1`. The viewer opens through Vite and loads `mypic1.ply`. Select another scene with a query string such as `?scene=/scenes/mypic2.ply`.

The 2.08 GB `point_cloud_last.ply`, raw videos, COLMAP outputs, downloadable COLMAP/FFmpeg tools, and cloud training caches are deliberately excluded from the portable zip. They are not needed for the portfolio viewer and would make transfer and verification impractical. Keep the original drive copy as the archival source for those training artifacts.
