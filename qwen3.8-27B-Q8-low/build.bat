@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
cl /nologo /O2 /EHsc /std:c++17 path_tracer.cpp /Fe:path_tracer.exe
