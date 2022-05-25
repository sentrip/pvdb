#include "pvdb_config.h"

layout (binding = PVDB_BINDING_CAMERA) uniform CameraUniform {
    mat4  mvp;
    mat4  proj;
	vec4  frustum_planes[6];
	ivec4 offset;
} uCamera;


bool uCameraFrustumContainsBox(vec3 pos, float size)
{
    const vec3 bmax = pos + vec3(size);

    float a = 1.0f;
    for (int i = 0; i < 6 && a >= 0.0f; ++i) {
        bvec3 b = greaterThan(uCamera.frustum_planes[i].xyz, vec3(0));
		vec3 n = mix(pos, bmax, b);
		a = dot(vec4(n, 1.0f), uCamera.frustum_planes[i]);
	}
	return (a >= 0.0);
}


float LightingPhong(vec3 to_light, vec3 normal, float min_brightness)
{
    return max(min_brightness, dot(normal, normalize(to_light)));

}

vec3 LightingGammaCorrection(vec3 color, float gamma_correction)
{
    return pow(color, vec3(1.0/gamma_correction));
}
