if (TARGET opencv)
    return()
endif()

message(STATUS "Third-party (external): creating target 'opencv'...")

include(FetchContent)
FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        3.4.12
    )

# opencv options
set(BUILD_opencv_python2 OFF CACHE BOOL "")
set(BUILD_opencv_python3 OFF CACHE BOOL "")
set(BUILD_PERF_TESTS OFF CACHE BOOL "")
set(BUILD_TESTS OFF CACHE BOOL "")
set(BUILD_DOCS OFF CACHE BOOL "")
set(BUILD_EXAMPLES OFF CACHE BOOL "")
set(BUILD_opencv_apps OFF CACHE BOOL "")
set(WITH_FFMPEG ON CACHE BOOL "")
set(WITH_GSTREAMER OFF CACHE BOOL "")
set(WITH_V4L ON CACHE BOOL "")
set(OPENCV_ENABLE_NONFREE ON CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")

# Disable problematic modules that cause compiler crashes
set(BUILD_opencv_dnn OFF CACHE BOOL "")
set(BUILD_opencv_objdetect OFF CACHE BOOL "")
set(BUILD_opencv_stitching OFF CACHE BOOL "")
set(BUILD_opencv_superres OFF CACHE BOOL "")
set(BUILD_opencv_videostab OFF CACHE BOOL "")
set(BUILD_opencv_ml OFF CACHE BOOL "")

# Reduce compiler optimization to avoid crashes
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -DNDEBUG" CACHE STRING "")
set(CMAKE_C_FLAGS_RELEASE "-O1 -DNDEBUG" CACHE STRING "")

FetchContent_MakeAvailable(opencv)
