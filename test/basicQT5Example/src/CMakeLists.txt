add_library(mainwindow mainwindow.cpp)
target_link_libraries (mainwindow Qt5::Widgets)
add_executable(MyProject main.cpp)
target_link_libraries (MyProject mainwindow)
