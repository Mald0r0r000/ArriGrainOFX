// MetalKernel.mm
// Arri Style Grain OFX Plugin — Objective-C++ Metal dispatch bridge
//
// RUNTIME COMPILATION: The Metal kernel source is embedded as a string
// constant and compiled at first use via [MTLDevice newLibraryWithSource:].
// This eliminates the need for a precompiled .metallib in the bundle.
// The Metal runtime caches compiled pipelines, so subsequent launches
// have near-zero overhead.
//
// NOTE: kKernelSource is the absolute single source of truth for the Metal shader.

#include "MetalKernel.h"
#import <Metal/Metal.h>
#import <simd/simd.h>

// ---------------------------------------------------------------------------
// Embedded Metal kernel source (from ArriGrainKernel.metal)
// ---------------------------------------------------------------------------
static NSString *const kKernelSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GrainParams {
    int   width;
    int   height;
    int   boundsX;
    int   boundsY;
    float time;
    int   srcRowBytes;
    int   dstRowBytes;
    int   formatSelect;
    int   stockSelect;
    int   processSelect;
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

    // --- Film Emulation Science ---
    float toe[3];
    float gamma[3];
    float shoulder[3];
    float crosstalk[9];
    float grainSize[3];
    float grainCorr[2]; // RG, GB correlation

    // --- Precalculated Performance Parameters ---
    float format_scale;
    float contrast_mod;
    float bw_mode;
};

// =====================================================================
// OPENSIMPLEX2S NOISE 3D
// Adapté pour Metal GPU — pas d'état global, pas de table de permutation
// Utilise un hash entier PCG pour les gradients
// =====================================================================

// Hash PCG3D — qualité statistique supérieure au hash 0.1031
inline uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// Gradient 3D depuis un hash entier — 12 gradients sur les arêtes du cube
inline float grad3d(uint hash, float x, float y, float z) {
    uint h = hash & 15u;
    float u = (h < 8u)  ? x : y;
    float v = (h < 4u)  ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) ? -u : u) + ((h & 2u) ? -v : v);
}

// OpenSimplex2S 3D
// Retourne une valeur dans [-1, 1], centrée sur 0
inline float simplex3d(float x, float y, float z, float seed) {
    // Skew vers la grille simplexe
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;
    
    float s = (x + y + z) * F3;
    int i = (int)floor(x + s);
    int j = (int)floor(y + s);
    int k = (int)floor(z + s);
    
    float t = (float)(i + j + k) * G3;
    float x0 = x - ((float)i - t);
    float y0 = y - ((float)j - t);
    float z0 = z - ((float)k - t);
    
    // Déterminer le simplexe
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }
    
    float x1 = x0 - (float)i1 + G3;
    float y1 = y0 - (float)j1 + G3;
    float z1 = z0 - (float)k1 + G3;
    float x2 = x0 - (float)i2 + 2.0f*G3;
    float y2 = y0 - (float)j2 + 2.0f*G3;
    float z2 = z0 - (float)k2 + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3;
    float y3 = y0 - 1.0f + 3.0f*G3;
    float z3 = z0 - 1.0f + 3.0f*G3;
    
    // Seed intégré dans le hash via offset entier
    int si = (int)(seed * 1000.0f);
    
    // Contributions des 4 coins
    float n = 0.0f;
    
    float t0 = 0.6f - x0*x0 - y0*y0 - z0*z0;
    if (t0 > 0.0f) {
        uint3 h0 = pcg3d(uint3(i + si, j, k));
        t0 *= t0;
        n += t0*t0 * grad3d(h0.x, x0, y0, z0);
    }
    float t1 = 0.6f - x1*x1 - y1*y1 - z1*z1;
    if (t1 > 0.0f) {
        uint3 h1 = pcg3d(uint3(i+i1 + si, j+j1, k+k1));
        t1 *= t1;
        n += t1*t1 * grad3d(h1.x, x1, y1, z1);
    }
    float t2 = 0.6f - x2*x2 - y2*y2 - z2*z2;
    if (t2 > 0.0f) {
        uint3 h2 = pcg3d(uint3(i+i2 + si, j+j2, k+k2));
        t2 *= t2;
        n += t2*t2 * grad3d(h2.x, x2, y2, z2);
    }
    float t3 = 0.6f - x3*x3 - y3*y3 - z3*z3;
    if (t3 > 0.0f) {
        uint3 h3 = pcg3d(uint3(i+1 + si, j+1, k+1));
        t3 *= t3;
        n += t3*t3 * grad3d(h3.x, x3, y3, z3);
    }
    
    // Normaliser vers [-1, 1]
    return clamp(n * 32.0f, -1.0f, 1.0f);
}

inline float grain_mask(float luma, float shadow_r, float highlight_r) {
    // Courbe en cloche asymétrique centrée sur les ombres
    float peak = 0.22f;
    float width_low  = 0.20f;  // flanc gauche (vers les noirs)
    float width_high = 0.45f;  // flanc droit (vers les hautes lumières)
    
    float width = (luma < peak) ? width_low : width_high;
    float bell = exp(-((luma - peak) * (luma - peak)) 
                    / (2.0f * width * width));
    
    // Plancher dans les noirs — le grain ne disparaît jamais totalement
    float floor_val = 0.12f * shadow_r;
    
    // Plafond dans les hautes lumières contrôlé par highlightResponse
    float ceiling = 1.0f - (luma * (1.0f - highlight_r));
    
    return clamp(max(floor_val, bell) * ceiling, 0.0f, 1.0f);
}

// ===== FILM RESPONSE & CROSS-TALK =====

inline float apply_hd_curve(float x, float toe, float gamma, float shoulder) {
    float safe_x = max(0.0f, x);
    if (safe_x <= toe) return safe_x;
    return shoulder - (shoulder - toe) * exp(-pow(safe_x - toe, gamma));
}

inline float3 apply_crosstalk(float3 c, constant float* mx) {
    return float3(
        c.r * mx[0] + c.g * mx[1] + c.b * mx[2],
        c.r * mx[3] + c.g * mx[4] + c.b * mx[5],
        c.r * mx[6] + c.g * mx[7] + c.b * mx[8]
    );
}

inline float blend_soft_light(float a, float b) {
    return (1.0f - 2.0f * b) * (a * a) + 2.0f * b * a;
}

inline float mix_f(float a, float b, float t) {
    return a + t * (b - a);
}

// ===== MAIN COMPUTE KERNEL =====

kernel void grainKernel(
    device const float* src       [[ buffer(0) ]],
    device       float* dst       [[ buffer(1) ]],
    constant GrainParams& params  [[ buffer(2) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    if ((int)gid.x >= params.width || (int)gid.y >= params.height) return;

    int imgX = (int)gid.x + params.boundsX;
    int imgY = (int)gid.y + params.boundsY;

    int srcPPR = params.srcRowBytes / (int)sizeof(float);
    int dstPPR = params.dstRowBytes / (int)sizeof(float);
    int si = (int)gid.y * srcPPR + (int)gid.x * 4;
    int di = (int)gid.y * dstPPR + (int)gid.x * 4;

    float R = src[si], G = src[si+1], B = src[si+2], A = src[si+3];

    // ----- 1. COLOR SCIENCE: H&D Curves & Cross-talk -----
    float r_hdr = apply_hd_curve(R, params.toe[0], params.gamma[0], params.shoulder[0]);
    float g_hdr = apply_hd_curve(G, params.toe[1], params.gamma[1], params.shoulder[1]);
    float b_hdr = apply_hd_curve(B, params.toe[2], params.gamma[2], params.shoulder[2]);
    
    float luma = 0.2126f*R + 0.7152f*G + 0.0722f*B;
    float luma_resp = apply_hd_curve(luma, params.toe[0], params.gamma[0], params.shoulder[0]);

    float r_resp = mix_f(r_hdr, luma_resp, params.bw_mode);
    float g_resp = mix_f(g_hdr, luma_resp, params.bw_mode);
    float b_resp = mix_f(b_hdr, luma_resp, params.bw_mode);
    
    float3 color_resp = apply_crosstalk(float3(r_resp, g_resp, b_resp), params.crosstalk);
    r_resp = color_resp.r; g_resp = color_resp.g; b_resp = color_resp.b;

    // Luma raw for grain mask & size modulation
    float luma_raw = 0.2126f*r_resp + 0.7152f*g_resp + 0.0722f*b_resp;

    // ----- 2. GRAIN GENERATION -----
    float format_scale = params.format_scale;

    // Modulate grain size in shadows [QUAL-4]
    float shadow_size_boost = 1.0f + 
        (1.0f - clamp(luma_raw / 0.4f, 0.0f, 1.0f)) 
        * 0.35f * params.shadowResponse;

    // Scale noise independent per channel
    float final_scale_R = max(0.1f, format_scale * params.grainSize[0] * params.resScale * shadow_size_boost);
    float final_scale_G = max(0.1f, format_scale * params.grainSize[1] * params.resScale * shadow_size_boost);
    float final_scale_B = max(0.1f, format_scale * params.grainSize[2] * params.resScale * shadow_size_boost);

    float zt_base = params.time * params.animSpeed;
    float px = float(imgX), py = float(imgY);

    float pxR = px / final_scale_R, pyR = py / final_scale_R;
    float pxG = px / final_scale_G, pyG = py / final_scale_G;
    float pxB = px / final_scale_B, pyB = py / final_scale_B;

    float oR = 0.0f, oG = 521.3f, oB = 194.2f;

    // Differential temporal octaves [QUAL-5]
    float zt_fine   = zt_base * 2.0f;
    float zt_medium = zt_base * 0.7f;
    float zt_coarse = zt_base * 0.25f;

    // Center + 2 diagonals spatial taps [QUAL-7] (limited to 3 taps for cost since PERF-1 texture is removed)
    float blur_r = params.grainSoftness * final_scale_R * 0.3f;
    float blur_g = params.grainSoftness * final_scale_G * 0.3f;
    float blur_b = params.grainSoftness * final_scale_B * 0.3f;

    // Function macro to compute 3-tap blurred simplex noise
    #define SIMPLEX_3TAP(PX, PY, ZT, SEED, BLUR) \
        ((simplex3d(PX, PY, ZT, SEED) + \
          simplex3d(PX + BLUR, PY + BLUR, ZT, SEED) + \
          simplex3d(PX - BLUR, PY - BLUR, ZT, SEED)) * 0.333333f * 0.5f)

    // Fine
    float nr1 = SIMPLEX_3TAP(pxR, pyR, zt_fine+oR, 0.0f);
    float ng1 = SIMPLEX_3TAP(pxG, pyG, zt_fine+oG, 10.0f);
    float nb1 = SIMPLEX_3TAP(pxB, pyB, zt_fine+oB, 20.0f);

    // Medium
    float ms=2.0f;
    float nr2 = SIMPLEX_3TAP(pxR/ms, pyR/ms, zt_medium+oR, 30.0f);
    float ng2 = SIMPLEX_3TAP(pxG/ms, pyG/ms, zt_medium+oG, 40.0f);
    float nb2 = SIMPLEX_3TAP(pxB/ms, pyB/ms, zt_medium+oB, 50.0f);

    // Coarse
    float cs=4.0f;
    float nr3 = SIMPLEX_3TAP(pxR/cs, pyR/cs, zt_coarse+oR, 60.0f);
    float ng3 = SIMPLEX_3TAP(pxG/cs, pyG/cs, zt_coarse+oG, 70.0f);
    float nb3 = SIMPLEX_3TAP(pxB/cs, pyB/cs, zt_coarse+oB, 80.0f);

    // Layer mixing
    float nr = nr1*params.mixFine + nr2*params.mixMedium + nr3*params.mixCoarse;
    float ng = ng1*params.mixFine + ng2*params.mixMedium + ng3*params.mixCoarse;
    float nb = nb1*params.mixFine + nb2*params.mixMedium + nb3*params.mixCoarse;

    // Correlation (Cross-channel grain mixing)
    float noiseR = nr;
    float noiseG = mix_f(ng, nr, params.grainCorr[0]);
    float noiseB = mix_f(nb, ng, params.grainCorr[1]);

    // RGB bias sliders
    noiseR *= params.biasR; 
    noiseG *= params.biasG; 
    noiseB *= params.biasB;

    // Contrast: grain_depth slider × process modifier
    float contrast_mod = params.contrast_mod;
    
    float ct = (1.0f+(params.grainDepth*2.0f)) * contrast_mod;
    noiseR=clamp(noiseR*ct,-0.5f,0.5f);
    noiseG=clamp(noiseG*ct,-0.5f,0.5f);
    noiseB=clamp(noiseB*ct,-0.5f,0.5f);

    // [QUAL-3] Correct Luma Masking (bell curve in shadows, ceiling in highlights)
    float im = grain_mask(luma_raw, params.shadowResponse, params.highlightResponse) * params.globalAmt;
    float nrn=mix_f(0.5f, noiseR+0.5f, im);
    float ngn=mix_f(0.5f, noiseG+0.5f, im);
    float nbn=mix_f(0.5f, noiseB+0.5f, im);

    if (params.showMask == 1) { 
        dst[di]   = nrn; 
        dst[di+1] = ngn; 
        dst[di+2] = nbn; 
        dst[di+3] = A; 
        return; 
    }

    dst[di]   = blend_soft_light(r_resp, nrn);
    dst[di+1] = blend_soft_light(g_resp, ngn);
    dst[di+2] = blend_soft_light(b_resp, nbn);
    dst[di+3] = A;
}
)MSL";

#include <os/lock.h>

// ---------------------------------------------------------------------------
// CPU-side PCG3D random generation for noise texture
// ---------------------------------------------------------------------------
static inline float pcg3d_scalar(uint32_t x, uint32_t y, uint32_t z) {
    simd_uint3 v = simd_make_uint3(x % 64, y % 64, z % 64);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return (float)v.x * (1.0f / (float)0xffffffffu);
}

// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> s_Pipeline = nil;
static id<MTLDevice> s_Device = nil;
static id<MTLTexture> s_NoiseTex = nil;
static os_unfair_lock s_PipelineLock = OS_UNFAIR_LOCK_INIT;

static bool ensurePipeline(id<MTLDevice> device) {
  os_unfair_lock_lock(&s_PipelineLock);

  if (s_Pipeline && s_Device == device) {
    os_unfair_lock_unlock(&s_PipelineLock);
    return true;
  }

  bool success = false;
  @autoreleasepool {
    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    opts.languageVersion = MTLLanguageVersion2_4;

    id<MTLLibrary> lib = [device newLibraryWithSource:kKernelSource
                                              options:opts
                                                error:&error];
    if (!lib) {
      NSLog(@"[ArriGrain] Metal compile error: %@", error);
      os_unfair_lock_unlock(&s_PipelineLock);
      return false;
    }

    id<MTLFunction> func = [lib newFunctionWithName:@"grainKernel"];
    if (!func) {
      NSLog(@"[ArriGrain] grainKernel function not found");
      os_unfair_lock_unlock(&s_PipelineLock);
      return false;
    }

    s_Pipeline = [device newComputePipelineStateWithFunction:func
                                                       error:&error];
    if (!s_Pipeline) {
      NSLog(@"[ArriGrain] Pipeline error: %@", error);
      os_unfair_lock_unlock(&s_PipelineLock);
      return false;
    }

    if (!s_NoiseTex || s_Device != device) {
      MTLTextureDescriptor *texDesc = [[MTLTextureDescriptor alloc] init];
      texDesc.textureType = MTLTextureType3D;
      texDesc.pixelFormat = MTLPixelFormatR32Float;
      texDesc.width = 64;
      texDesc.height = 64;
      texDesc.depth = 64;
      texDesc.mipmapLevelCount = 1;
      texDesc.usage = MTLTextureUsageShaderRead;
      s_NoiseTex = [device newTextureWithDescriptor:texDesc];
        
      float *noiseData = (float*)malloc(64 * 64 * 64 * sizeof(float));
      for (uint32_t z = 0; z < 64; z++) {
        for (uint32_t y = 0; y < 64; y++) {
          for (uint32_t x = 0; x < 64; x++) {
            noiseData[z*64*64 + y*64 + x] = pcg3d_scalar(x, y, z);
          }
        }
      }
      MTLRegion region = MTLRegionMake3D(0, 0, 0, 64, 64, 64);
      [s_NoiseTex replaceRegion:region mipmapLevel:0 slice:0 withBytes:noiseData bytesPerRow:64 * sizeof(float) bytesPerImage:64 * 64 * sizeof(float)];
      free(noiseData);
    }

    s_Device = device;
    NSLog(@"[ArriGrain] Metal pipeline ready (threads/group: %lu)",
          (unsigned long)s_Pipeline.maxTotalThreadsPerThreadgroup);
    success = true;
  }
  
  os_unfair_lock_unlock(&s_PipelineLock);
  return success;
}

// ---------------------------------------------------------------------------
// Public: dispatch compute kernel
// ---------------------------------------------------------------------------
int RunMetalKernel(void *p_CmdQueue, const void *p_Src, void *p_Dst,
                   const GrainParams *p_Params) {
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)p_CmdQueue;
    id<MTLDevice> device = queue.device;

    if (!ensurePipeline(device))
      return 2; // kOfxStatFailed

    // DaVinci Resolve provides id<MTLBuffer> when
    // kOfxImageEffectPropMetalRenderSupported is true. We must bridge cast
    // these directly rather than treating them as CPU memory pointers.
    id<MTLBuffer> srcBuf = (__bridge id<MTLBuffer>)p_Src;
    id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)p_Dst;

    if (!srcBuf || !dstBuf) {
      NSLog(@"[ArriGrain] Invalid Metal buffers provided by host");
      return 2;
    }

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:s_Pipeline];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:dstBuf offset:0 atIndex:1];
    // Use setBytes for small params instead of allocating a 3rd buffer
    [enc setBytes:p_Params length:sizeof(GrainParams) atIndex:2];

    NSUInteger w = s_Pipeline.threadExecutionWidth;
    NSUInteger h = s_Pipeline.maxTotalThreadsPerThreadgroup / w;
    MTLSize tgSize = MTLSizeMake(w, h, 1);
    MTLSize grid = MTLSizeMake((NSUInteger)p_Params->width,
                               (NSUInteger)p_Params->height, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tgSize];
    [enc endEncoding];

    [cmdBuf commit];
  }
  return 0; // kOfxStatOK
}
