@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /std:c++17 /EHsc /openmp path_tracer.cpp /Fe:pt.exe
echo EXITCODE=%ERRORLEVEL%
