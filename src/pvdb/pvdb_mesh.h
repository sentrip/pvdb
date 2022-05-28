//
// Created by Djordje on 5/25/2022.
//

#ifndef PVDB_MESH_H
#define PVDB_MESH_H

#include "pvdb_buffer.h"
#include "pvdb_allocator.h"


#ifndef PVDB_GLOBAL_MESHES
#define PVDB_GLOBAL_MESHES                  GlobalMeshes
#endif

#ifndef PVDB_GLOBAL_MESH_VERTICES
#define PVDB_GLOBAL_MESH_VERTICES           GlobalMeshVertices
#endif

#ifndef PVDB_GLOBAL_MESH_ALLOC
#define PVDB_GLOBAL_MESH_ALLOC              GlobalMeshAlloc
#endif

#define pvdb_meshes_at(a, addr)             pvdb_buf_at(a, PVDB_GLOBAL_MESHES, addr)
#define pvdb_mesh_vert_at(a, addr)          pvdb_buf_at(a, PVDB_GLOBAL_MESH_VERTICES, addr)


#define PVDB_MESH_LOG2DIM                   5u
#define PVDB_MESH_STRUCT_SIZE               5u
#define PVDB_MESH_MEMBER_COUNT              0u
#define PVDB_MESH_MEMBER_OFFSET_CAPACITY    1u
#define PVDB_MESH_MEMBER_XYZ                2u
#define PVDB_MESH_OFFSET_MASK               0xffffffu
#define PVDB_MESH_CAPACITY_MASK             0xffu
#define PVDB_MESH_CAPACITY_SHIFT            24u
#define PVDB_MESH_INITIAL_CAPACITY_LOG2DIM  9u
#define PVDB_MESH_MAX_CAPACITY_LOG2DIM      (3u + (3u * PVDB_MESH_LOG2DIM))
#define PVDB_MESH_VERTEX_ALLOC_LEVELS       (PVDB_MESH_MAX_CAPACITY_LOG2DIM - PVDB_MESH_INITIAL_CAPACITY_LOG2DIM)


#define PVDB_VERTEX_STRUCT_SIZE             2u
#define PVDB_VERTEX_POSITION_MASK           0xffu
#define PVDB_VERTEX_DIRECTION_MASK          0x7u
#define PVDB_VERTEX_DIRECTION_SHIFT         16u
#define PVDB_VERTEX_ROTATION_MASK           0x1fu
#define PVDB_VERTEX_ROTATION_SHIFT          19u
#define PVDB_VERTEX_AMBIENT_OCC_MASK        0xfu
#define PVDB_VERTEX_AMBIENT_OCC_SHIFT       24u


struct pvdb_mesh {
    uint  count;
    uint  offset_capacity;
    ivec3 xyz;
};

#define pvdb_mesh_offset(v)             ( (v) & PVDB_MESH_OFFSET_MASK )
#define pvdb_mesh_capacity_log2dim(v)   ( ( (v) >> PVDB_MESH_CAPACITY_SHIFT ) & PVDB_MESH_CAPACITY_MASK )
#define pvdb_meshes_data_offset(m, i)   ( ( (m) * PVDB_MESH_STRUCT_SIZE ) + (i) )


struct pvdb_vertex {
    // Voxel     - 0b01111111'11111111'11111111'11111111
    // UNUSED    - 0b10000000'00000000'00000000'00000000
    uint voxel;
    // Position  - 0b00000000'00000000'11111111'11111111
    // Direction - 0b00000000'00000111'00000000'00000000
    // Rotation  - 0b00000000'11111000'00000000'00000000
    // AmbiOccl  - 0b11111111'00000000'00000000'00000000
    uint info;
};


#define pvdb_vertex_xyz_index(v)        ( (v).info & PVDB_VERTEX_POSITION_MASK )
#define pvdb_vertex_direction(v)        ( ( (v).info >> PVDB_VERTEX_DIRECTION_SHIFT ) & PVDB_VERTEX_DIRECTION_MASK )
#define pvdb_vertex_rotation(v)         ( ( (v).info >> PVDB_VERTEX_ROTATION_SHIFT ) & PVDB_VERTEX_ROTATION_MASK )
#define pvdb_vertex_ambient_occ(v)      ( ( (v).info >> PVDB_VERTEX_AMBIENT_OCC_SHIFT ) & PVDB_VERTEX_AMBIENT_OCC_MASK )
#define pvdb_vertex_xyz(v)              pvdb_index_to_coord(pvdb_vertex_xyz_index(v), PVDB_MESH_LOG2DIM)

PVDB_INLINE pvdb_vertex
pvdb_vertex_make(
    uint                    voxel,
    uint                    pos,
    uint                    direction,
    uint                    rotation,
    uint                    ambient_occ)
{
    pvdb_vertex v;
    v.voxel = voxel;
    v.info = pos
        | (direction << PVDB_VERTEX_DIRECTION_SHIFT)
        | (rotation << PVDB_VERTEX_ROTATION_SHIFT)
        | (ambient_occ << PVDB_VERTEX_AMBIENT_OCC_SHIFT);
    return v;
}


/// Read
PVDB_INLINE ivec3
pvdb_mesh_get_xyz(
    pvdb_buf_in             meshes,
    uint                    mesh)
{
    const uint data_offset = pvdb_meshes_data_offset(mesh, PVDB_MESH_MEMBER_XYZ);
    return ivec3(
        int(pvdb_meshes_at(meshes, data_offset)),
        int(pvdb_meshes_at(meshes, data_offset + 1u)),
        int(pvdb_meshes_at(meshes, data_offset + 2u))
    );
}


PVDB_INLINE pvdb_mesh
pvdb_mesh_get(
    pvdb_buf_in             meshes,
    uint                    mesh)
{
    pvdb_mesh m;
    const uint data_offset = pvdb_meshes_data_offset(mesh, 0u);
    m.count = pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_COUNT);
    m.offset_capacity = pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_OFFSET_CAPACITY);
    m.xyz.x = int(pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ));
    m.xyz.y = int(pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ + 1u));
    m.xyz.z = int(pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ + 2u));
    return m;
}


PVDB_INLINE pvdb_vertex
pvdb_mesh_read(
    pvdb_buf_in             vertices,
    uint                    index)
{
    pvdb_vertex v;
    const uint data_offset = index * PVDB_VERTEX_STRUCT_SIZE;
    v.voxel = pvdb_mesh_vert_at(vertices, data_offset);
    v.info = pvdb_mesh_vert_at(vertices, data_offset + 1u);
    return v;
}


/// Write
PVDB_INLINE void
pvdb_mesh_write(
    pvdb_buf_inout          vertices,
    uint                    index,
    PVDB_IN(pvdb_vertex)    v)
{
    const uint data_offset = index * PVDB_VERTEX_STRUCT_SIZE;
    pvdb_mesh_vert_at(vertices, data_offset) = v.voxel;
    pvdb_mesh_vert_at(vertices, data_offset + 1u) = v.info;
}


PVDB_INLINE uint
pvdb_mesh_reallocate_if_necessary(
    pvdb_buf_inout          meshes,
    pvdb_buf_inout          mesh_alloc,
    uint                    mesh,
    uint                    index)
{
    const uint data_offset = pvdb_meshes_data_offset(mesh, PVDB_MESH_MEMBER_OFFSET_CAPACITY);
    for (;;) {
        const uint cap_l2d = pvdb_meshes_at(meshes, data_offset);
        const uint offset = pvdb_mesh_offset(cap_l2d);
        const uint capacity_l2d = pvdb_mesh_capacity_log2dim(cap_l2d);

        // the currently allocated mesh has enough space
        if (capacity_l2d > 0u && index < (1u << capacity_l2d))
            return offset;

        // get new capacity log2dim to fit at least index
        uint new_capacity_l2d = capacity_l2d > 0u ? capacity_l2d : PVDB_MESH_INITIAL_CAPACITY_LOG2DIM;
        while (index >= (1u << new_capacity_l2d)) new_capacity_l2d += 1;

        // allocate new mesh
        const uint allocator_index = capacity_l2d > 0u ? new_capacity_l2d - PVDB_MESH_INITIAL_CAPACITY_LOG2DIM : 0u;
        const uint new_offset = pvdb_allocator_alloc(mesh_alloc, allocator_index);
        const uint new_cap_l2d = new_offset | (new_capacity_l2d << PVDB_MESH_CAPACITY_SHIFT);

        // We swap both offset and capacity in a single word to ensure that the entire allocation state is swapped atomically
        if (atomicCompSwap(pvdb_meshes_at(meshes, data_offset), cap_l2d, new_cap_l2d) == cap_l2d) {
            if (capacity_l2d > 0u) {
                PVDB_PRINTF("\n\t MESH COPY: mesh: %u, old_offset: %u, size: %u\n", mesh, offset, 1u << capacity_l2d);
                // enqueue copy from offset to new offset of size capacity
            }
            return new_offset;
        }
        // If we failed to swap we must free the mesh for the next attempt
        else {
            pvdb_allocator_free(meshes, allocator_index, new_offset);
        }
    }
    return 0u;
}


/// add a mesh and return index
PVDB_INLINE uint
pvdb_mesh_add(
    pvdb_buf_inout          meshes,
    pvdb_buf_inout          mesh_alloc,
    PVDB_IN(ivec3)          xyz)
{
    const uint level = 0;
    const uint mesh = pvdb_allocator_alloc(mesh_alloc, PVDB_MESH_VERTEX_ALLOC_LEVELS + level);
    const uint data_offset = pvdb_meshes_data_offset(mesh, 0u);
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_COUNT) = 0u;
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_OFFSET_CAPACITY) = 0u;
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ) = uint(xyz.x);
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ + 1u) = uint(xyz.y);
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_XYZ + 2u) = uint(xyz.z);
    return mesh;
}


/// remove the given mesh
PVDB_INLINE void
pvdb_mesh_remove(
    pvdb_buf_inout          meshes,
    pvdb_buf_inout          mesh_alloc,
    uint                    mesh)
{
    const uint level = 0;
    const uint data_offset = pvdb_meshes_data_offset(mesh, 0u);
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_COUNT) = 0u;
    pvdb_meshes_at(meshes, data_offset + PVDB_MESH_MEMBER_OFFSET_CAPACITY) = 0u;
    pvdb_allocator_free(mesh_alloc, PVDB_MESH_VERTEX_ALLOC_LEVELS + level, mesh);
}


/// add voxel face to the given mesh
PVDB_INLINE void
pvdb_mesh_add_face(
    pvdb_buf_inout          meshes,
    pvdb_buf_inout          mesh_alloc,
    pvdb_buf_inout          vertices,
    uint                    mesh,
    PVDB_IN(pvdb_vertex)    v)
{
    const uint index = atomicAdd(pvdb_meshes_at(meshes, pvdb_meshes_data_offset(mesh, PVDB_MESH_MEMBER_COUNT)), 1u);
    const uint offset = pvdb_mesh_reallocate_if_necessary(meshes, mesh_alloc, mesh, index);
    pvdb_mesh_write(vertices, offset + index, v);
}


#endif //PVDB_MESH_H
