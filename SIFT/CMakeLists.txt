cmake_minimum_required (VERSION 3.8)
set(PROJECT_NAME Sift)
project(${PROJECT_NAME})

#输出路径
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

#D:/eigen/eigen路径下存放C++矩阵运算的Eigen 
set(OPENCV_INCLUDE "$ENV{OPENCV_DIR}/newbuild/install/include;$ENV{OPENCV_DIR}/newbuild/install/include/opencv2;$ENV{OPENCV_DIR}/newbuild/install/include/opencv;D:/eigen/eigen;$ENV{KINECTSDK20_DIR}/inc;")

#让CMake找到头文件
include_directories(${OPENCV_INCLUDE})   


#让CMake找到源文件
aux_source_directory(src DIR_SRCS)  
file(GLOB_RECURSE HEADERS *.h)
source_group("Header Files" FILES ${HEADERS})

#添加一个可执行文件构建目标
add_executable(${PROJECT_NAME} ${DIR_SRCS} ${HEADERS})

#链接库D:\opencv\newbuild\install\x64\vc14\lib
set(OPENCV_LIB_DIRS $ENV{OPENCV_DIR}/newbuild/install/x64/vc14/lib)
#GLOB选项将会为所有匹配查询表达式的文件生成一个文件list，并将该list存储进变量LIBS里
file(GLOB LIBSD "${OPENCV_LIB_DIRS}/*330d.lib")
file(GLOB LIBS "${OPENCV_LIB_DIRS}/*330.lib")
file(GLOB LIBKINECTS "$ENV{KINECTSDK20_DIR}/Lib/x64/*.lib")

target_link_libraries(${PROJECT_NAME} debug ${LIBSD})
target_link_libraries(${PROJECT_NAME} optimized ${LIBS})
target_link_libraries(${PROJECT_NAME} ${LIBKINECTS})