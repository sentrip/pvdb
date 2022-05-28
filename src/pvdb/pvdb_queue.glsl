#ifndef PVDB_QUEUE_BINDING
#error Must define PVDB_QUEUE_BINDING
#endif

#ifndef PVDB_QUEUE_ARRAY_SIZE
#define PVDB_QUEUE_ARRAY_SIZE 1u
#endif

#ifndef PVDB_GLOBAL_QUEUE
#define PVDB_GLOBAL_QUEUE               GlobalQueue
#endif

#define PVDB_QUEUE_HEADER_ONLY
#include "pvdb_queue.h"

layout(std430, binding = PVDB_QUEUE_BINDING) buffer PVDB_QUEUE_NAME {
    pvdb_queue_header header;
    PVDB_QUEUE_TYPE data[];
} PVDB_GLOBAL_QUEUE[PVDB_QUEUE_ARRAY_SIZE];

#define PVDB_QUEUE_IMPLEMENTATION
#include "pvdb_queue.h"

#undef PVDB_QUEUE_BINDING
#undef PVDB_QUEUE_ARRAY_SIZE
