//
// Created by Djordje on 5/24/2022.
//

#ifndef PVDB_RAYCAST_H
#define PVDB_RAYCAST_H

#include "pvdb_tree_read.h"
#include "pvdb_math.h"


#define PVDB_RAYCAST_MAX_ITER   256


struct pvdb_ray_hit {
    vec3    normal;
    float   t;
    uint    voxel;
};


PVDB_INLINE bool
pvdb_raycast_leaf(
    pvdb_tree_in            tree,
    PVDB_IN(pvdb_ray)       ray,
    PVDB_OUT(pvdb_ray_hit)  hit,
    uint                    node,
    PVDB_IN(ivec3)          node_pos,
    PVDB_IN(vec3)           t)
{
    const int dim = int(pvdb_dim(tree, 0));

    pvdb_dda dda;
    pvdb_dda_set_from_ray(dda, ray, t);
    pvdb_dda_prepare_level(dda, node_pos, vec3(1.0f));

    uint iter = 0;
    while (iter++ < PVDB_RAYCAST_MAX_ITER
           && dda.p.x >= 0 && dda.p.y >= 0 && dda.p.z >= 0
           && dda.p.x < dim && dda.p.y < dim && dda.p.z < dim)
   {
        const uint i = pvdb_coord_local_to_index(tree, dda.p, 0);
        if (pvdb_read_node_mask(tree, node, i)) {
            hit.voxel = pvdb_read_node(tree, 0, node, i);
            hit.t = dda.t.x;
            hit.normal = pvdb_ray_hit_normal(ray.dir, ray.pos + ray.dir * dda.t.x, dda.p + node_pos);
			return true;
        }
        pvdb_dda_next(dda);
        pvdb_dda_step(dda);
    }
    return false;
}


PVDB_INLINE bool
pvdb_raycast(
    pvdb_tree_in            tree,
    PVDB_IN(pvdb_ray)       ray,
    PVDB_OUT(pvdb_ray_hit)  hit,
    PVDB_INOUT(uint)        level,
    PVDB_INOUT(ivec3)       node_pos,
    PVDB_ARRAY_INOUT(ivec3, offset, PVDB_MAX_LEVELS),
    PVDB_ARRAY_INOUT(uint,  nodes, PVDB_MAX_LEVELS),
    PVDB_ARRAY_INOUT(float, t_max, PVDB_MAX_LEVELS))
{
    const uint root = pvdb_root(tree);
    vec3 t_start = vec3(0.0f, PVDB_DDA_FLT_MAX, 0.0f);
    vec3 bbox_max = vec3(ivec3(int(pvdb_total_dim(tree, root))));
    if (!pvdb_ray_box_intersect(ray, vec3(0.0f), bbox_max, t_start.x, t_start.y)) {
        return false;
    }

    uint index, node = nodes[level];
    t_start.x += PVDB_DDA_FLT_MIN;
	t_max[root] = t_start.y - PVDB_DDA_FLT_MIN;

    vec3 voxel_dim[PVDB_MAX_LEVELS];
    for (uint i = 0; i < pvdb_levels(tree); ++i)
        voxel_dim[i] = vec3(ivec3(int(pvdb_total_dim_voxel(tree, i))));

    pvdb_dda dda;
    pvdb_dda_set_from_ray(dda, ray, t_start);
    pvdb_dda_prepare_level(dda, node_pos, voxel_dim[level]);

    uint iter = 0;
    while (level <= root && iter++ < PVDB_RAYCAST_MAX_ITER && dda.p.x >= 0 && dda.p.y >= 0 && dda.p.z >= 0) {
        const int dim = int(pvdb_dim(tree, level));
        if (dda.p.x >= dim && dda.p.y >= dim && dda.p.z >= dim)
            break;

        pvdb_dda_next(dda);

        index = pvdb_coord_local_to_index(tree, dda.p, level);

        if (pvdb_read_node_mask(tree, node, index)) {
//            PVDB_PRINTF("READ: level: %u, node: %u, index: %u, xyz: (%d, %d, %d) - HIT\n", level, node, index, dda.p.x, dda.p.y, dda.p.z);
            const uint leaf = pvdb_read_node(tree, level, node, index);
            dda.t.x += PVDB_DDA_FLT_MIN;
            node_pos = pvdb_access_xyz(tree, node_pos, level+1) + (dda.p << int(pvdb_total_log2dim_voxel(tree, level)));

            if (pvdb_is_tile(leaf)) {
                hit.t = dda.t.x;
                hit.normal = pvdb_ray_hit_normal(ray.dir, ray.pos + ray.dir * dda.t.x, node_pos);
                hit.voxel = node & PVDB_NODE;
                return true;
            }
            else if (level == 1) {
                if (pvdb_raycast_leaf(tree, ray, hit, leaf, node_pos, dda.t))
                    return true;
                pvdb_dda_step(dda);
            }
            else {
                node = leaf;
                nodes[--level] = leaf;
                t_max[level] = dda.t.y - PVDB_DDA_FLT_MIN;
                pvdb_dda_prepare_level(dda, node_pos, voxel_dim[level]);
            }
        }
        else {
//            PVDB_PRINTF("READ: level: %u, node: %u, index: %u, xyz: (%d, %d, %d) - MISS\n", level, node, index, dda.p.x, dda.p.y, dda.p.z);
            pvdb_dda_step(dda);
        }

        while(level <= root && dda.t.x >= t_max[level]) {
            ++level;
            if (level > root) break;
            node = nodes[level];
            dda.t.x += PVDB_DDA_FLT_MIN;
            node_pos = pvdb_access_xyz(tree, node_pos, level+1);
            pvdb_dda_prepare_level(dda, node_pos, voxel_dim[level]);
        }
    }
    return false;
}


PVDB_INLINE bool
pvdb_raycast(
    pvdb_tree_in            tree,
    PVDB_IN(pvdb_ray)       ray,
    PVDB_OUT(pvdb_ray_hit)  hit,
    PVDB_IN(ivec3)          offset)
{
    ivec3 node_pos = ivec3(0);
    uint level = pvdb_root(tree);
    uint nodes[PVDB_MAX_LEVELS];
    float t_max[PVDB_MAX_LEVELS];
    nodes[level] = 0u;
    ivec3 offset_per_level[PVDB_MAX_LEVELS];
    pvdb_coord_global_to_coords(tree, offset, offset_per_level);
//    for (uint i = 1; i <= level; ++i) offset_per_level[i] += offset_per_level[i-1];
    return pvdb_raycast(tree, ray, hit, level, node_pos, offset_per_level, nodes, t_max);
}


#endif //PVDB_RAYCAST_H
