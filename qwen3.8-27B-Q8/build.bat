@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "E:\dev\pt-ai-benchmark\qwen3.8-27B-Q8"
cl /nologo /O2 /std:c++17 /EHsc /MT /DNDEBUG pathtracer.cpp /Fe:pathtracer.exe
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK
