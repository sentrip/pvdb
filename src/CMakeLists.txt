
function(pvdb_copy_gpu_sources_to_target_dir TARGET)
    add_custom_command(TARGET ${TARGET} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/src/pvdb $<TARGET_FILE_DIR:${TARGET}>/pvdb)
endfunction()

add_subdirectory(gpu)
add_subdirectory(cvdb)
add_subdirectory(test)

add_executable(sample main.cpp)
target_link_libraries(sample PRIVATE pvdb_gpu_debug_device pvdb)
pvdb_copy_gpu_sources_to_target_dir(sample)
pvdb_copy_sdl_to_target_dir(sample)
