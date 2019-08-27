# STACCATO
STACCATO: STefAn's Computational vibroaCoustics Analysis TOol

Finite-Element based software tool including an interactive geometric 
modeling kernel (using Open CASCADE Technology).

![STACCATO](http://sicklinger.com/images/STACCATO_02.png)
![STACCATO](http://sicklinger.com/images/STACCATO_03.png)
![STACCATO](http://sicklinger.com/images/STACCATO_00.png)
![STACCATO](http://sicklinger.com/images/STACCATO_01.png)



# Building
Run cmake on Win64
cmake .. -G "Visual Studio 14 2015 Win64"


# QT 5 deployment 
cd C:\software\repos\STACCATO\bin64\Release
C:\software\libs\QT\Qt591\5.9.1\msvc2015_64\bin\windeployqt.exe STACCATO.exe