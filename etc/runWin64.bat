echo Setting OpenCASCADE
call C:\software\libs\OpenCASCADE7.1.0-vc12-64\opencascade-7.1.0\env.bat vc12 win64 Release
echo Setting QT 5.7.0
set "PATH=C:\software\libs\QT\QT570\5.7\msvc2013_64\bin;%PATH%"
echo Starting STACCATO ...
call ping -n 3 127.0.0.1 > NUL
"..\bin64\Release\STACCATO.exe"
