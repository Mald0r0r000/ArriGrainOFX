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
    float grainSize[3];
    float grainCorr[2];
    float format_scale;
    float contrast_mod;
};

// =====================================================================
// OPENSIMPLEX2S NOISE 3D
// =====================================================================

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

inline float grad3d(uint hash, float x, float y, float z) {
    uint h = hash & 15u;
    float u = (h < 8u)  ? x : y;
    float v = (h < 4u)  ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) ? -u : u) + ((h & 2u) ? -v : v);
}

inline float simplex3d(float x, float y, float z, float seed) {
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;
    
    float s = (x + y + z) * F3;
    int i = int(floor(x + s));
    int j = int(floor(y + s));
    int k = int(floor(z + s));
    
    float t = float(i + j + k) * G3;
    float x0 = x - (float(i) - t);
    float y0 = y - (float(j) - t);
    float z0 = z - (float(k) - t);
    
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
    
    float x1 = x0 - float(i1) + G3;
    float y1 = y0 - float(j1) + G3;
    float z1 = z0 - float(k1) + G3;
    float x2 = x0 - float(i2) + 2.0f*G3;
    float y2 = y0 - float(j2) + 2.0f*G3;
    float z2 = z0 - float(k2) + 2.0f*G3;
    float x3 = x0 - 1.0f + 3.0f*G3;
    float y3 = y0 - 1.0f + 3.0f*G3;
    float z3 = z0 - 1.0f + 3.0f*G3;
    
    int si = int(seed * 1000.0f);
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
    return clamp(n * 32.0f, -1.0f, 1.0f);
}

// ===== MAIN COMPUTE KERNEL =====
kernel void grainKernel(
    device const float* src       [[ buffer(0) ]],
    device       float* dst       [[ buffer(1) ]],
    constant GrainParams& params  [[ buffer(2) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    if (int(gid.x) >= params.width || int(gid.y) >= params.height) return;

    int imgX = int(gid.x) + params.boundsX;
    int imgY = int(gid.y) + params.boundsY;

    int srcPPR = params.srcRowBytes / 4;
    int dstPPR = params.dstRowBytes / 4;
    int si = int(gid.y) * srcPPR + int(gid.x) * 4;
    int di = int(gid.y) * dstPPR + int(gid.x) * 4;

    float R = src[si], G = src[si+1], B = src[si+2], A = src[si+3];

    // Test mode
    if (params.showMask == 5) {
        dst[di]   = 1.0f;
        dst[di+1] = 0.0f;
        dst[di+2] = 0.0f;
        dst[di+3] = 1.0f;
        return;
    }

    // ----- 1. DENSITY-DRIVEN AMPLITUDE -----
    float densR = clamp(R * 1.5f, 0.0f, 1.0f);
    float densG = clamp(G * 1.5f, 0.0f, 1.0f);
    float densB = clamp(B * 1.5f, 0.0f, 1.0f);

    float shiftR = (params.shadowResponse - 0.5f) * 0.3f;
    float hlScale = params.highlightResponse;

    float dR = densR - shiftR;
    float dG = densG - shiftR;
    float dB = densB - shiftR;

    float grainAmpR = clamp(dR * (1.0f - dR) * 4.0f * (1.0f + (hlScale - 1.0f) * dR), 0.0f, 1.0f);
    float grainAmpG = clamp(dG * (1.0f - dG) * 4.0f * (1.0f + (hlScale - 1.0f) * dG), 0.0f, 1.0f);
    float grainAmpB = clamp(dB * (1.0f - dB) * 4.0f * (1.0f + (hlScale - 1.0f) * dB), 0.0f, 1.0f);

    // ----- 2. GRAIN GENERATION -----
    float luma = 0.2126f*R + 0.7152f*G + 0.0722f*B;
    
    float diag = sqrt(float(params.width * params.width + params.height * params.height));
    float res_norm = diag / 4420.0f;
    float base_scale = params.format_scale * params.resScale * res_norm;

    float shadow_boost = 1.0f + (1.0f - clamp(luma / 0.4f, 0.0f, 1.0f)) * 0.35f * params.shadowResponse;
    
    float scaleR = max(0.1f, base_scale * params.grainSize[0] * shadow_boost);
    float scaleG = max(0.1f, base_scale * params.grainSize[1] * shadow_boost);
    float scaleB = max(0.1f, base_scale * params.grainSize[2] * shadow_boost);

    float px = float(imgX), py = float(imgY);
    float zt = params.time * params.animSpeed;
    
    float zt_fine = zt * 2.0f;
    float zt_med  = zt * 0.7f;
    float zt_coarse = zt * 0.25f;

    float blurR = params.grainSoftness * scaleR * 0.3f;
    float blurG = params.grainSoftness * scaleG * 0.3f;
    float blurB = params.grainSoftness * scaleB * 0.3f;

    // Fine grain with softness (3-tap)
    float nR1 = (simplex3d(px/scaleR, py/scaleR, zt_fine, 0.0f) +
                 simplex3d((px+blurR)/scaleR, (py+blurR)/scaleR, zt_fine, 0.0f) +
                 simplex3d((px-blurR)/scaleR, (py-blurR)/scaleR, zt_fine, 0.0f)) * 0.333f;
    float nG1 = (simplex3d(px/scaleG, py/scaleG, zt_fine, 10.0f) +
                 simplex3d((px+blurG)/scaleG, (py+blurG)/scaleG, zt_fine, 10.0f) +
                 simplex3d((px-blurG)/scaleG, (py-blurG)/scaleG, zt_fine, 10.0f)) * 0.333f;
    float nB1 = (simplex3d(px/scaleB, py/scaleB, zt_fine, 20.0f) +
                 simplex3d((px+blurB)/scaleB, (py+blurB)/scaleB, zt_fine, 20.0f) +
                 simplex3d((px-blurB)/scaleB, (py-blurB)/scaleB, zt_fine, 20.0f)) * 0.333f;

    // Medium grain
    float nR2 = simplex3d(px/(scaleR*2.0f), py/(scaleR*2.0f), zt_med, 30.0f);
    float nG2 = simplex3d(px/(scaleG*2.0f), py/(scaleG*2.0f), zt_med, 40.0f);
    float nB2 = simplex3d(px/(scaleB*2.0f), py/(scaleB*2.0f), zt_med, 50.0f);

    // Coarse grain (envelope)
    float nR3 = simplex3d(px/(scaleR*4.0f), py/(scaleR*4.0f), zt_coarse, 60.0f);
    float nG3 = simplex3d(px/(scaleG*4.0f), py/(scaleG*4.0f), zt_coarse, 70.0f);
    float nB3 = simplex3d(px/(scaleB*4.0f), py/(scaleB*4.0f), zt_coarse, 80.0f);

    // Clumping: coarse as multiplicative envelope
    float envR = 1.0f + nR3 * params.mixCoarse * 0.6f;
    float envG = 1.0f + nG3 * params.mixCoarse * 0.6f;
    float envB = 1.0f + nB3 * params.mixCoarse * 0.6f;

    float wsum = params.mixFine + params.mixMedium;
    float norm = 1.0f / max(wsum, 0.001f);
    
    float noiseR = (nR1*params.mixFine + nR2*params.mixMedium) * norm * envR;
    float noiseG = (nG1*params.mixFine + nG2*params.mixMedium) * norm * envG;
    float noiseB = (nB1*params.mixFine + nB2*params.mixMedium) * norm * envB;

    // Cross-channel correlation
    float corrRG = clamp(params.grainCorr[0], 0.0f, 1.0f);
    float corrGB = clamp(params.grainCorr[1], 0.0f, 1.0f);
    
    float grainG = noiseG + (noiseR - noiseG) * corrRG * 0.5f;
    float grainB = noiseB + (grainG - noiseB) * corrGB * 0.5f;
    float grainR = noiseR;

    // Apply bias and contrast
    grainR *= params.biasR;
    grainG *= params.biasG;
    grainB *= params.biasB;

    float ct = (1.0f + params.grainDepth * 2.0f) * params.contrast_mod;
    grainR = clamp(grainR * ct, -1.0f, 1.0f);
    grainG = clamp(grainG * ct, -1.0f, 1.0f);
    grainB = clamp(grainB * ct, -1.0f, 1.0f);

    // Debug modes
    if (params.showMask == 1) {
        dst[di]   = grainAmpR;
        dst[di+1] = grainAmpG;
        dst[di+2] = grainAmpB;
        dst[di+3] = A;
        return;
    }
    if (params.showMask == 2) {
        dst[di]   = grainR * 0.5f + 0.5f;
        dst[di+1] = grainG * 0.5f + 0.5f;
        dst[di+2] = grainB * 0.5f + 0.5f;
        dst[di+3] = A;
        return;
    }
    if (params.showMask == 3) {
        dst[di]   = grainAmpR;
        dst[di+1] = grainAmpR;
        dst[di+2] = grainAmpR;
        dst[di+3] = A;
        return;
    }
    if (params.showMask == 4) {
        dst[di]   = R;
        dst[di+1] = G;
        dst[di+2] = B;
        dst[di+3] = A;
        return;
    }

    // Additive blend in log space
    float logScale = 0.3f;
    float outR = R + grainR * grainAmpR * params.globalAmt * logScale;
    float outG = G + grainG * grainAmpG * params.globalAmt * logScale;
    float outB = B + grainB * grainAmpB * params.globalAmt * logScale;

    dst[di]   = outR;
    dst[di+1] = outG;
    dst[di+2] = outB;
    dst[di+3] = A;
}
)MSL";

#include <os/lock.h>



// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> s_Pipeline = nil;
static id<MTLDevice> s_Device = nil;

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

    // LOG: Print key parameters to diagnose
    static int logCounter = 0;
    if (logCounter++ % 60 == 0) {
      NSLog(@"[ArriGrain] === PARAMETER DUMP ===");
      NSLog(@"[ArriGrain] width=%d height=%d", p_Params->width, p_Params->height);
      NSLog(@"[ArriGrain] globalAmt=%.3f grainDepth=%.3f grainSoftness=%.3f",
            p_Params->globalAmt, p_Params->grainDepth, p_Params->grainSoftness);
      NSLog(@"[ArriGrain] mixFine=%.3f mixMedium=%.3f mixCoarse=%.3f",
            p_Params->mixFine, p_Params->mixMedium, p_Params->mixCoarse);
      NSLog(@"[ArriGrain] biasR=%.3f biasG=%.3f biasB=%.3f",
            p_Params->biasR, p_Params->biasG, p_Params->biasB);
      NSLog(@"[ArriGrain] shadowResponse=%.3f highlightResponse=%.3f",
            p_Params->shadowResponse, p_Params->highlightResponse);
      NSLog(@"[ArriGrain] format_scale=%.3f contrast_mod=%.3f",
            p_Params->format_scale, p_Params->contrast_mod);
      NSLog(@"[ArriGrain] grainSize=[%.3f, %.3f, %.3f]",
            p_Params->grainSize[0], p_Params->grainSize[1], p_Params->grainSize[2]);
      NSLog(@"[ArriGrain] grainCorr=[%.3f, %.3f]",
            p_Params->grainCorr[0], p_Params->grainCorr[1]);
      NSLog(@"[ArriGrain] showMask=%d", p_Params->showMask);
    }

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
