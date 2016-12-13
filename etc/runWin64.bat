echo Setting OpenCASCADE
call C:\software\libs\OpenCASCADE7.1.0-vc12-64\opencascade-7.1.0\env.bat vc12 win64 Release
echo Setting QT 5.6.1
set "PATH=C:\software\libs\QT\QT561\5.6\msvc2013_64\bin;%PATH%"
echo Starting STACCATO ...
"..\bin64\Release\STACCATO.exe"
