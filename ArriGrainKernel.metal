// ArriGrainKernel.metal
// Arri Style Grain OFX Plugin — Metal Compute Kernel
// Direct port of DCTL v9: Arri Style - Soft Light Blend Engine
//
// PRECISION STRATEGY:
//   - Noise hash internals: half (sufficient for [0,1] grain values, 2x throughput on M-series)
//   - Film response, blend, pixel I/O: float (preserves HDR/Log values above 1.0)
//
// DETERMINISM:
//   - Seed = frame_number (from kOfxPropTime) * anim_speed + per-channel/per-layer offsets
//   - No random state — identical output on every playback/export

#include <metal_stdlib>
using namespace metal;

// ----- Parameter struct (must match MetalKernel.h GrainParams) -----
struct GrainParams {
    int   width;
    int   height;
    int   srcWidth;
    float time;
    int   renderOffsetX;
    int   renderOffsetY;
    int   srcRowBytes;
    int   dstRowBytes;

    int   stockSelect;
    float resScale;
    float animSpeed;
    float mixFine;
    float mixMedium;
    float mixCoarse;
    float globalAmt;
    float grainDepth;
    float grainSoftness;
    float biasR;
    float biasG;
    float biasB;
    float shadowResponse;
    float highlightResponse;
    int   showMask;
};

// =====================================================================
// NOISE UTILITIES — half precision (sufficient for grain hash values)
// =====================================================================

inline half safe_fract_h(half x) {
    return x - floor(x);
}

inline half mix_h(half a, half b, half t) {
    return a + t * (b - a);
}

// 3D scalar hash — produces pseudo-random value in [0, 1]
inline half hash_3d_scalar(half x, half y, half z, half seed) {
    half3 p3 = half3(x, y, z);
    p3.x = safe_fract_h(p3.x * half(0.1031h));
    p3.y = safe_fract_h(p3.y * half(0.1031h));
    p3.z = safe_fract_h(p3.z * half(0.1031h));

    half dot_v = dot(p3, half3(p3.y + half(33.33h),
                               p3.z + half(33.33h),
                               p3.x + half(33.33h)));
    p3 += dot_v;
    return safe_fract_h((p3.x + p3.y) * p3.z + seed);
}

// 3D value noise — trilinear interpolation of hash values
// Returns float for downstream precision
inline float noise_3d(float xf, float yf, float zf, float seedf) {
    half x = half(xf);
    half y = half(yf);
    half z = half(zf);
    half seed = half(seedf);

    half ix = floor(x);  half iy = floor(y);  half iz = floor(z);
    half fx = safe_fract_h(x);
    half fy = safe_fract_h(y);
    half fz = safe_fract_h(z);

    // Hermite smoothstep
    half ux = fx * fx * (half(3.0h) - half(2.0h) * fx);
    half uy = fy * fy * (half(3.0h) - half(2.0h) * fy);
    half uz = fz * fz * (half(3.0h) - half(2.0h) * fz);

    // 8 corners of the unit cube
    half c000 = hash_3d_scalar(ix,            iy,            iz,            seed);
    half c100 = hash_3d_scalar(ix + half(1),  iy,            iz,            seed);
    half c010 = hash_3d_scalar(ix,            iy + half(1),  iz,            seed);
    half c110 = hash_3d_scalar(ix + half(1),  iy + half(1),  iz,            seed);
    half c001 = hash_3d_scalar(ix,            iy,            iz + half(1),  seed);
    half c101 = hash_3d_scalar(ix + half(1),  iy,            iz + half(1),  seed);
    half c011 = hash_3d_scalar(ix,            iy + half(1),  iz + half(1),  seed);
    half c111 = hash_3d_scalar(ix + half(1),  iy + half(1),  iz + half(1),  seed);

    // Trilinear interpolation
    half x00 = mix_h(c000, c100, ux);
    half x10 = mix_h(c010, c110, ux);
    half x01 = mix_h(c001, c101, ux);
    half x11 = mix_h(c011, c111, ux);

    half y0 = mix_h(x00, x10, uy);
    half y1 = mix_h(x01, x11, uy);

    return float(mix_h(y0, y1, uz));
}

// =====================================================================
// FILM RESPONSE & BLEND — float precision (HDR-safe)
// =====================================================================

// Luminance-driven grain response curve
inline float get_film_response(float luma, float shadow_r, float high_r) {
    float safe_luma = max(0.0f, luma);
    float response = pow(safe_luma, shadow_r * 2.5f);
    if (safe_luma > 0.35f) {
        float h = (safe_luma - 0.35f) * 1.55f;
        response = response * (1.0f - (h * (1.0f - high_r)));
    }
    return max(0.0f, response);
}

// Pegtop soft-light blend: (1 - 2b) * a² + 2b * a
// a = image pixel, b = grain (normalised 0-1, 0.5 = neutral)
// NO clamping on output → preserves HDR values > 1.0
inline float blend_soft_light(float a, float b) {
    return (1.0f - 2.0f * b) * (a * a) + 2.0f * b * a;
}

inline float mix_f(float a, float b, float t) {
    return a + t * (b - a);
}

// =====================================================================
// MAIN COMPUTE KERNEL
// =====================================================================

kernel void grainKernel(
    device const float* src       [[ buffer(0) ]],
    device       float* dst       [[ buffer(1) ]],
    constant GrainParams& params  [[ buffer(2) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    // Bounds check
    if ((int)gid.x >= params.width || (int)gid.y >= params.height)
        return;

    // Pixel coordinates in full image space (account for tiled rendering offset)
    int imgX = (int)gid.x + params.renderOffsetX;
    int imgY = (int)gid.y + params.renderOffsetY;

    // Calculate byte offsets (rowBytes may include padding)
    int srcPixelsPerRow = params.srcRowBytes / (int)sizeof(float);
    int dstPixelsPerRow = params.dstRowBytes / (int)sizeof(float);

    int srcIdx = (int)gid.y * srcPixelsPerRow + (int)gid.x * 4;
    int dstIdx = (int)gid.y * dstPixelsPerRow + (int)gid.x * 4;

    // Read RGBA (float — HDR safe)
    float R = src[srcIdx + 0];
    float G = src[srcIdx + 1];
    float B = src[srcIdx + 2];
    float A = src[srcIdx + 3];

    // ----- 1. STOCK SCALE -----
    float base_scale = 1.0f;
    if (params.stockSelect == 0)      base_scale = 0.5f;   // 50D — fine
    else if (params.stockSelect == 1) base_scale = 1.0f;   // 250D — medium
    else if (params.stockSelect == 2) base_scale = 1.8f;   // 500T — coarse

    float final_scale = base_scale * params.resScale;
    if (final_scale < 0.1f) final_scale = 0.1f;

    // ----- 2. LUMINANCE ANALYSIS -----
    float luma = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float drive_factor = get_film_response(luma, params.shadowResponse, params.highlightResponse);

    if (params.showMask == 1) {
        dst[dstIdx + 0] = drive_factor;
        dst[dstIdx + 1] = drive_factor;
        dst[dstIdx + 2] = drive_factor;
        dst[dstIdx + 3] = A;
        return;
    }

    // ----- 3. COORDINATES & TIME -----
    float z_time = params.time * params.animSpeed;
    float px = float(imgX) / final_scale;
    float py = float(imgY) / final_scale;

    // Per-channel seed offsets (avoid correlated RGB noise)
    float offset_r = 0.0f;
    float offset_g = 521.3f;
    float offset_b = 194.2f;

    // ----- 4. MULTI-OCTAVE 3D NOISE -----

    // Layer FINE (high frequency)
    float f_z = z_time * 2.0f;
    float nr_1 = noise_3d(px, py, f_z + offset_r,  0.0f) - 0.5f;
    float ng_1 = noise_3d(px, py, f_z + offset_g, 10.0f) - 0.5f;
    float nb_1 = noise_3d(px, py, f_z + offset_b, 20.0f) - 0.5f;

    // Layer MEDIUM
    float m_sc = 2.0f;
    float nr_2 = noise_3d(px / m_sc, py / m_sc, z_time + offset_r, 30.0f) - 0.5f;
    float ng_2 = noise_3d(px / m_sc, py / m_sc, z_time + offset_g, 40.0f) - 0.5f;
    float nb_2 = noise_3d(px / m_sc, py / m_sc, z_time + offset_b, 50.0f) - 0.5f;

    // Layer COARSE (low frequency, large structure)
    float c_sc = 4.0f;
    float nr_3 = noise_3d(px / c_sc, py / c_sc, z_time + offset_r, 60.0f) - 0.5f;
    float ng_3 = noise_3d(px / c_sc, py / c_sc, z_time + offset_g, 70.0f) - 0.5f;
    float nb_3 = noise_3d(px / c_sc, py / c_sc, z_time + offset_b, 80.0f) - 0.5f;

    // Mix layers
    float nr = nr_1 * params.mixFine + nr_2 * params.mixMedium + nr_3 * params.mixCoarse;
    float ng = ng_1 * params.mixFine + ng_2 * params.mixMedium + ng_3 * params.mixCoarse;
    float nb = nb_1 * params.mixFine + nb_2 * params.mixMedium + nb_3 * params.mixCoarse;

    // RGB bias
    nr *= params.biasR;
    ng *= params.biasG;
    nb *= params.biasB;

    // Softness & contrast
    float softness_factor = 1.0f - (params.grainSoftness * 0.5f);
    nr *= softness_factor;
    ng *= softness_factor;
    nb *= softness_factor;

    float contrast = 1.0f + (params.grainDepth * 2.0f);
    nr = clamp(nr * contrast, -0.5f, 0.5f);
    ng = clamp(ng * contrast, -0.5f, 0.5f);
    nb = clamp(nb * contrast, -0.5f, 0.5f);

    // ----- 5. SOFT LIGHT APPLICATION -----
    float intensity_map = drive_factor * params.globalAmt;

    // Normalise grain to [0, 1] for blend (0.5 = neutral)
    // Modulate by intensity_map (luminance response × global amount)
    float nr_norm = mix_f(0.5f, nr + 0.5f, intensity_map);
    float ng_norm = mix_f(0.5f, ng + 0.5f, intensity_map);
    float nb_norm = mix_f(0.5f, nb + 0.5f, intensity_map);

    // Final blend — NO clamp → HDR/Log safe
    float r_out = blend_soft_light(R, nr_norm);
    float g_out = blend_soft_light(G, ng_norm);
    float b_out = blend_soft_light(B, nb_norm);

    // Write output (alpha pass-through)
    dst[dstIdx + 0] = r_out;
    dst[dstIdx + 1] = g_out;
    dst[dstIdx + 2] = b_out;
    dst[dstIdx + 3] = A;
}
