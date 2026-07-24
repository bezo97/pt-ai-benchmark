@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d e:\dev\pt-ai-benchmark\Hy3-IQ1_M
cl /O2 /EHsc /std:c++17 pathtracer.cpp
if exist pt.exe ( pt.exe )
