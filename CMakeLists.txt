cmake_minimum_required(VERSION 3.14)
project(Locus CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

add_executable(
  ${PROJECT_NAME}
  main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${VENDOR_DIR}/include)

target_link_libraries(${PROJECT_NAME} PRIVATE
    debug ${VENDOR_DIR}/lib/opencv_world4120d.lib
    optimized ${VENDOR_DIR}/lib/opencv_world4120.lib
)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "$<IF:$<CONFIG:Debug>,${VENDOR_DIR}/bin/opencv_world4120d.dll,${VENDOR_DIR}/bin/opencv_world4120.dll>"
  $<TARGET_FILE_DIR:${PROJECT_NAME}>
)