
add_library(pvdb_gpu STATIC
    Backend.cpp
    Context.cpp
    Objects.cpp
    Pass.cpp)


add_library(pvdb_gpu_debug_device STATIC DebugDevice.cpp)
target_link_libraries(pvdb_gpu_debug_device PUBLIC pvdb_gpu)

find_package(Vulkan REQUIRED)
target_link_libraries(pvdb_gpu PUBLIC Vulkan::Vulkan)

include(ExternalProject)
ExternalProject_Add(
    tinyvk
    GIT_REPOSITORY https://github.com/sentrip/tinyvk.git
    BUILD_COMMAND ""
    CMAKE_ARGS -D CMAKE_BUILD_TYPE=$<CONFIG> -D CMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -D TINYVK_BUILD_TESTS=OFF -D TINYVK_HEADER_ONLY=ON)

ExternalProject_Get_property(tinyvk INSTALL_DIR)
target_include_directories(pvdb_gpu PUBLIC ${INSTALL_DIR}/include)

# TODO: Fix updating tinyvk every build
#add_dependencies(pvdb_gpu tinyvk)

include(tinyvk)
tinyvk_link_shaderc(pvdb_gpu)
tinyvk_link_spirv_cross(pvdb_gpu)

# VULKAN_SDK_PATH
file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" VULKAN_SDK_VAR)
target_compile_definitions(pvdb_gpu PRIVATE VULKAN_SDK_PATH="${VULKAN_SDK_VAR}")

# SDL2
target_include_directories(pvdb_gpu_debug_device PUBLIC $ENV{VULKAN_SDK}/Include/SDL2)
target_link_libraries(pvdb_gpu_debug_device PUBLIC $ENV{VULKAN_SDK}/Lib/SDL2.lib)

function(pvdb_copy_sdl_to_target_dir TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                       $ENV{VULKAN_SDK}/Bin/SDL2.dll $<TARGET_FILE_DIR:${TARGET}>/SDL2.dll)
endfunction()
