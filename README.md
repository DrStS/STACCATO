#---------------------------------------------------------------------#
# STACCATO
STACCATO: STefAn's Computational vibroaCoustics Analysis TOol
#---------------------------------------------------------------------#
Finite-Element based software tool including an interactive geometric 
modeling kernel (using Open CASCADE Technology).
#---------------------------------------------------------------------#
![STACCATO](http://sicklinger.com/images/SPILOT03.png)
![STACCATO](http://sicklinger.com/images/STACCATO.png)
![STACCATO](http://sicklinger.com/images/SPILOT01.png)
![STACCATO](http://sicklinger.com/images/SPILOT02.png)
#---------------------------------------------------------------------#
# Building
Run cmake on Win64
```
cmake .. -G "Visual Studio 14 2015 Win64"
```
#---------------------------------------------------------------------#
# QT 5 deployment
``` 
cd C:\software\repos\STACCATO\bin64\Release
C:\software\libs\QT\Qt591\5.9.1\msvc2015_64\bin\windeployqt.exe STACCATO.exe
```