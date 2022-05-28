

add_library(pvdb STATIC
        Runtime.cpp
        objects/Camera.cpp
        objects/Allocator.cpp
        objects/Tree.cpp
        objects/Queue.cpp)

target_link_libraries(pvdb PRIVATE pvdb_gpu)


function(pvdb_fix_macros TARGET)
    if (MSVC)
        target_compile_options(${TARGET} PUBLIC "/Zc:preprocessor")
    endif()
endfunction()

pvdb_fix_macros(pvdb)
