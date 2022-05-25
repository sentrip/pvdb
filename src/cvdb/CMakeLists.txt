

add_library(pvdb STATIC
        Runtime.cpp
        Camera.cpp
        Allocator.cpp
        Tree.cpp)

target_link_libraries(pvdb PRIVATE pvdb_gpu)
