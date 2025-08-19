cmake_minimum_required(VERSION 3.16)
project(face_recognition_opencv_project)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置跨平台路径处理
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 OpenVINO (可选，如果找不到则使用纯 OpenCV)
find_package(OpenVINO QUIET)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件
set(SOURCES
    src/main.cpp
    src/face_detection.cpp
    src/face_recognition.cpp
    src/utils.cpp
    src/face_manager.cpp
    src/recognition_engine.cpp
)

# 创建主可执行文件
add_executable(face_recognition ${SOURCES})

# 链接库
target_link_libraries(face_recognition
    ${OpenCV_LIBS}
)

# 如果找到 OpenVINO，添加支持（虽然不再需要，但保留兼容性）
if(OpenVINO_FOUND)
    target_link_libraries(face_recognition
        openvino::runtime
    )
    target_link_libraries(enroll_photos
        openvino::runtime
    )
    add_definitions(-DUSE_OPENVINO)
    message(STATUS "OpenVINO found, enabling OpenVINO support")
else()
    message(STATUS "OpenVINO not found, using OpenCV only")
endif()

# 设置输出目录
set_target_properties(face_recognition PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制必要的资源文件到输出目录
add_custom_command(TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models
    ${CMAKE_BINARY_DIR}/bin/models
    COMMENT "Copying models directory to output"
)

add_custom_command(TARGET face_recognition POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/pictures
    ${CMAKE_BINARY_DIR}/bin/pictures
    COMMENT "Copying pictures directory to output"
)


