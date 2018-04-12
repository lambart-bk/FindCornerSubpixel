
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )


add_executable(findcornersubpixel findcornersubpixel.cpp)
target_link_libraries(findcornersubpixel ${OpenCV_LIBRARIES} )


