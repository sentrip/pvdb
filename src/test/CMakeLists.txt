

add_executable(pvdb_test
    tests.cpp
    # TODO: Find out how to compile both PVDB_ALLOCATOR_MASK and PVDB_ALLOCATOR_DEBUG
#    test_pvdb_allocator.cpp
#    test_pvdb_tree_read.cpp
#    test_pvdb_tree_write.cpp
#    test_pvdb_raycast.cpp
    test_pvdb_mesh.cpp
    )
pvdb_copy_gpu_sources_to_target_dir(pvdb_test)

add_executable(pvdb_gpu_test
    tests.cpp
    test_gpu_pvdb_tree_write.cpp
    test_gpu_pvdb_queue.cpp
    )

target_link_libraries(pvdb_gpu_test PRIVATE pvdb_gpu_debug_device pvdb)
pvdb_copy_gpu_sources_to_target_dir(pvdb_gpu_test)
pvdb_copy_sdl_to_target_dir(pvdb_gpu_test)
pvdb_fix_macros(pvdb_gpu_test)
