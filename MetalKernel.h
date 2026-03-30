// MetalKernel.h
// Arri Style Grain OFX Plugin — Metal dispatch bridge header
// C-linkage header shared between ArriGrainPlugin.cpp and MetalKernel.mm

#ifndef METAL_KERNEL_H
#define METAL_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// Must match the struct in ArriGrainKernel.metal
typedef struct {
  int width;
  int height;
  int boundsX;
  int boundsY;
  float time; // Frame number from kOfxPropTime (deterministic)
  int srcRowBytes;
  int dstRowBytes;

  // --- Parameters ---
  int formatSelect;
  int processSelect;
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
  int showMask;

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
} GrainParams;

// Returns 0 on success (kOfxStatOK)
int RunMetalKernel(void *p_CmdQueue, const void *p_Src, void *p_Dst,
                   const GrainParams *p_Params);

#ifdef __cplusplus
}
#endif

#endif // METAL_KERNEL_H
