// MetalKernel.mm
// Arri Style Grain OFX Plugin — Objective-C++ Metal dispatch bridge
//
// RUNTIME COMPILATION: The Metal kernel source is embedded as a string
// constant and compiled at first use via [MTLDevice newLibraryWithSource:].
// This eliminates the need for a precompiled .metallib in the bundle.
// The Metal runtime caches compiled pipelines, so subsequent launches
// have near-zero overhead.

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
};

// ===== NOISE (float precision to avoid spatial grid artifacts for px > 1024) =====

inline float safe_fract_f(float x) { return x - floor(x); }
inline float mix_f2(float a, float b, float t) { return a + t * (b - a); }

inline float hash_3d_scalar(float x, float y, float z, float seed) {
    float3 p3 = float3(x, y, z);
    p3.x = safe_fract_f(p3.x * 0.1031f);
    p3.y = safe_fract_f(p3.y * 0.1031f);
    p3.z = safe_fract_f(p3.z * 0.1031f);
    float dot_v = dot(p3, float3(p3.y + 33.33f, p3.z + 33.33f, p3.x + 33.33f));
    p3 += dot_v;
    return safe_fract_f((p3.x + p3.y) * p3.z + seed);
}

inline float noise_3d(float x, float y, float z, float seed) {
    float ix = floor(x), iy = floor(y), iz = floor(z);
    float fx = safe_fract_f(x), fy = safe_fract_f(y), fz = safe_fract_f(z);
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    float uz = fz * fz * (3.0f - 2.0f * fz);
    float c000 = hash_3d_scalar(ix, iy, iz, seed);
    float c100 = hash_3d_scalar(ix+1.0f, iy, iz, seed);
    float c010 = hash_3d_scalar(ix, iy+1.0f, iz, seed);
    float c110 = hash_3d_scalar(ix+1.0f, iy+1.0f, iz, seed);
    float c001 = hash_3d_scalar(ix, iy, iz+1.0f, seed);
    float c101 = hash_3d_scalar(ix+1.0f, iy, iz+1.0f, seed);
    float c011 = hash_3d_scalar(ix, iy+1.0f, iz+1.0f, seed);
    float c111 = hash_3d_scalar(ix+1.0f, iy+1.0f, iz+1.0f, seed);
    float x00 = mix_f2(c000, c100, ux);
    float x10 = mix_f2(c010, c110, ux);
    float x01 = mix_f2(c001, c101, ux);
    float x11 = mix_f2(c011, c111, ux);
    float y0 = mix_f2(x00, x10, uy);
    float y1 = mix_f2(x01, x11, uy);
    return mix_f2(y0, y1, uz);
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
    float r_resp = R, g_resp = G, b_resp = B;
    if (params.stockSelect == 4) { // Double-X B&W
        float luma = 0.2126f*R + 0.7152f*G + 0.0722f*B;
        float luma_resp = apply_hd_curve(luma, params.toe[0], params.gamma[0], params.shoulder[0]);
        r_resp = g_resp = b_resp = luma_resp;
    } else {
        r_resp = apply_hd_curve(R, params.toe[0], params.gamma[0], params.shoulder[0]);
        g_resp = apply_hd_curve(G, params.toe[1], params.gamma[1], params.shoulder[1]);
        b_resp = apply_hd_curve(B, params.toe[2], params.gamma[2], params.shoulder[2]);
    }
    
    float3 color_resp = apply_crosstalk(float3(r_resp, g_resp, b_resp), params.crosstalk);
    r_resp = color_resp.r; g_resp = color_resp.g; b_resp = color_resp.b;

    // Luma drive for grain blending (more visible in midtones/shadows)
    float luma_drive = 0.2126f*r_resp + 0.7152f*g_resp + 0.0722f*b_resp;

    // ----- 2. GRAIN GENERATION -----
    float format_scale = 1.0f;
    if (params.formatSelect == 0) format_scale = 4.0f;        // 8mm
    else if (params.formatSelect == 1) format_scale = 2.0f;   // 16mm
    else if (params.formatSelect == 2) format_scale = 1.0f;   // 35mm
    else if (params.formatSelect == 3) format_scale = 0.4f;   // 70mm / IMAX

    // Scale noise independent per channel
    float final_scale_R = max(0.1f, format_scale * params.grainSize[0] * params.resScale);
    float final_scale_G = max(0.1f, format_scale * params.grainSize[1] * params.resScale);
    float final_scale_B = max(0.1f, format_scale * params.grainSize[2] * params.resScale);

    float zt = params.time * params.animSpeed;
    float px = float(imgX), py = float(imgY);

    float pxR = px / final_scale_R, pyR = py / final_scale_R;
    float pxG = px / final_scale_G, pyG = py / final_scale_G;
    float pxB = px / final_scale_B, pyB = py / final_scale_B;

    float oR = 0.0f, oG = 521.3f, oB = 194.2f;

    // Fine
    float fz = zt*2.0f;
    float nr1=noise_3d(pxR, pyR, fz+oR, 0.0f)-0.5f;
    float ng1=noise_3d(pxG, pyG, fz+oG, 10.0f)-0.5f;
    float nb1=noise_3d(pxB, pyB, fz+oB, 20.0f)-0.5f;

    // Medium
    float ms=2.0f;
    float nr2=noise_3d(pxR/ms, pyR/ms, zt+oR, 30.0f)-0.5f;
    float ng2=noise_3d(pxG/ms, pyG/ms, zt+oG, 40.0f)-0.5f;
    float nb2=noise_3d(pxB/ms, pyB/ms, zt+oB, 50.0f)-0.5f;

    // Coarse
    float cs=4.0f;
    float nr3=noise_3d(pxR/cs, pyR/cs, zt+oR, 60.0f)-0.5f;
    float ng3=noise_3d(pxG/cs, pyG/cs, zt+oG, 70.0f)-0.5f;
    float nb3=noise_3d(pxB/cs, pyB/cs, zt+oB, 80.0f)-0.5f;

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

    float sf = 1.0f-(params.grainSoftness*0.5f);
    noiseR*=sf; noiseG*=sf; noiseB*=sf;
    
    // Contrast: grain_depth slider × process modifier
    float contrast_mod = 1.0f;
    if (params.processSelect == 1) contrast_mod = 1.5f;       // Bleach Bypass
    else if (params.processSelect == 2) contrast_mod = 1.2f;  // Reversal
    
    float ct = (1.0f+(params.grainDepth*2.0f)) * contrast_mod;
    noiseR=clamp(noiseR*ct,-0.5f,0.5f);
    noiseG=clamp(noiseG*ct,-0.5f,0.5f);
    noiseB=clamp(noiseB*ct,-0.5f,0.5f);

    float im = luma_drive * params.globalAmt;
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

// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> s_Pipeline = nil;
static id<MTLDevice> s_Device = nil;
static dispatch_once_t s_OnceToken;

static bool ensurePipeline(id<MTLDevice> device) {
  if (s_Pipeline && s_Device == device)
    return true;

  __block bool success = false;
  dispatch_once(&s_OnceToken, ^{
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
        return;
      }

      id<MTLFunction> func = [lib newFunctionWithName:@"grainKernel"];
      if (!func) {
        NSLog(@"[ArriGrain] grainKernel function not found");
        return;
      }

      s_Pipeline = [device newComputePipelineStateWithFunction:func
                                                         error:&error];
      if (!s_Pipeline) {
        NSLog(@"[ArriGrain] Pipeline error: %@", error);
        return;
      }

      s_Device = device;
      NSLog(@"[ArriGrain] Metal pipeline ready (threads/group: %lu)",
            (unsigned long)s_Pipeline.maxTotalThreadsPerThreadgroup);
      success = true;
    }
  });

  return s_Pipeline != nil;
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
