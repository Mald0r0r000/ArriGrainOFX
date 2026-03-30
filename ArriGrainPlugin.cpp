// ArriGrainPlugin.cpp
// Arri Style Grain OFX Plugin — Host code (raw OFX C API)
//
// Single-file OFX plugin for DaVinci Resolve with Metal rendering.
// Registers parameters, handles describe/render actions, delegates
// pixel processing to the Metal kernel via MetalKernel.mm.
//
// DETERMINISTIC: Frame number from kOfxPropTime is the sole time seed.
// HDR-SAFE: Float pixel depth only, no clamping above 1.0.

#include <cstdio>
#include <cstdlib>
#include <cstring>

// OFX SDK headers
#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxProperty.h"

// Metal dispatch
#include "MetalKernel.h"

// ---------------------------------------------------------------------------
// Metal property defines (may not be in older OFX SDK headers)
// ---------------------------------------------------------------------------
#ifndef kOfxImageEffectPropMetalRenderSupported
#define kOfxImageEffectPropMetalRenderSupported                                \
  "OfxImageEffectPropMetalRenderSupported"
#endif
#ifndef kOfxImageEffectPropMetalEnabled
#define kOfxImageEffectPropMetalEnabled "OfxImageEffectPropMetalEnabled"
#endif
#ifndef kOfxImageEffectPropMetalCommandQueue
#define kOfxImageEffectPropMetalCommandQueue                                   \
  "OfxImageEffectPropMetalCommandQueue"
#endif

// OFX export macro (guard against SDK redefining it)
#ifdef OfxExport
#undef OfxExport
#endif
#ifdef _WIN32
#define OfxExport extern "C" __declspec(dllexport)
#else
#define OfxExport extern "C" __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------
// Plugin identity
// ---------------------------------------------------------------------------
#define PLUGIN_ID "com.arristyle.grain.metal"
#define PLUGIN_NAME "Arri Style Grain"
#define PLUGIN_GROUP "Film Emulation"
#define PLUGIN_VMAJ 1
#define PLUGIN_VMIN 0

// ---------------------------------------------------------------------------
// Parameter name constants
// ---------------------------------------------------------------------------
#define P_FORMAT "formatSelect"
#define P_STOCK "stockSelect"
#define P_PROCESS "processSelect"
#define P_RESSCALE "resScale"
#define P_SPEED "animSpeed"
#define P_FINE "mixFine"
#define P_MEDIUM "mixMedium"
#define P_COARSE "mixCoarse"
#define P_AMOUNT "globalAmt"
#define P_DEPTH "grainDepth"
#define P_SOFT "grainSoftness"
#define P_BIAS_R "biasR"
#define P_BIAS_G "biasG"
#define P_BIAS_B "biasB"
#define P_SHADOW "shadowResponse"
#define P_HIGHLIGHT "highlightResponse"
#define P_SHOWMASK "showMask"

// ---------------------------------------------------------------------------
// Globals: host + suites
// ---------------------------------------------------------------------------
static OfxHost *gHost = nullptr;
static OfxPropertySuiteV1 *gPropSuite = nullptr;
static OfxParameterSuiteV1 *gParamSuite = nullptr;
static OfxImageEffectSuiteV1 *gEffSuite = nullptr;

// ---------------------------------------------------------------------------
// Instance data (parameter handles cached per instance)
// ---------------------------------------------------------------------------
struct InstData {
  OfxParamHandle format, stock, process;
  OfxParamHandle resScale, speed;
  OfxParamHandle fine, medium, coarse;
  OfxParamHandle amount, depth, softness;
  OfxParamHandle biasR, biasG, biasB;
  OfxParamHandle shadow, highlight;
  OfxParamHandle showMask;
};

// ===== HELPER: define grouped double param =================================
static void defDouble(OfxParamSetHandle ps, const char *name, const char *label,
                      double def, double lo, double hi, double inc,
                      const char *hint, const char *grp) {
  OfxPropertySetHandle p;
  gParamSuite->paramDefine(ps, kOfxParamTypeDouble, name, &p);
  gPropSuite->propSetString(p, kOfxParamPropDoubleType, 0,
                            kOfxParamDoubleTypePlain);
  gPropSuite->propSetString(p, kOfxPropLabel, 0, label);
  gPropSuite->propSetDouble(p, kOfxParamPropDefault, 0, def);
  gPropSuite->propSetDouble(p, kOfxParamPropMin, 0, lo);
  gPropSuite->propSetDouble(p, kOfxParamPropMax, 0, hi);
  gPropSuite->propSetDouble(p, kOfxParamPropDisplayMin, 0, lo);
  gPropSuite->propSetDouble(p, kOfxParamPropDisplayMax, 0, hi);
  gPropSuite->propSetDouble(p, kOfxParamPropIncrement, 0, inc);
  gPropSuite->propSetInt(p, kOfxParamPropAnimates, 0, 1);
  if (hint)
    gPropSuite->propSetString(p, kOfxParamPropHint, 0, hint);
  if (grp)
    gPropSuite->propSetString(p, kOfxParamPropParent, 0, grp);
}

static void defInt(OfxParamSetHandle ps, const char *name, const char *label,
                   int def, int lo, int hi, const char *hint, const char *grp) {
  OfxPropertySetHandle p;
  gParamSuite->paramDefine(ps, kOfxParamTypeInteger, name, &p);
  gPropSuite->propSetString(p, kOfxPropLabel, 0, label);
  gPropSuite->propSetInt(p, kOfxParamPropDefault, 0, def);
  gPropSuite->propSetInt(p, kOfxParamPropMin, 0, lo);
  gPropSuite->propSetInt(p, kOfxParamPropMax, 0, hi);
  gPropSuite->propSetInt(p, kOfxParamPropDisplayMin, 0, lo);
  gPropSuite->propSetInt(p, kOfxParamPropDisplayMax, 0, hi);
  gPropSuite->propSetInt(p, kOfxParamPropAnimates, 0, 1);
  if (hint)
    gPropSuite->propSetString(p, kOfxParamPropHint, 0, hint);
  if (grp)
    gPropSuite->propSetString(p, kOfxParamPropParent, 0, grp);
}

static void defBool(OfxParamSetHandle ps, const char *name, const char *label,
                    bool def, const char *hint, const char *grp) {
  OfxPropertySetHandle p;
  gParamSuite->paramDefine(ps, kOfxParamTypeBoolean, name, &p);
  gPropSuite->propSetString(p, kOfxPropLabel, 0, label);
  gPropSuite->propSetInt(p, kOfxParamPropDefault, 0, def ? 1 : 0);
  gPropSuite->propSetInt(p, kOfxParamPropAnimates, 0, 0);
  if (hint)
    gPropSuite->propSetString(p, kOfxParamPropHint, 0, hint);
  if (grp)
    gPropSuite->propSetString(p, kOfxParamPropParent, 0, grp);
}

static void defGroup(OfxParamSetHandle ps, const char *name, const char *label,
                     bool open) {
  OfxPropertySetHandle p;
  gParamSuite->paramDefine(ps, kOfxParamTypeGroup, name, &p);
  gPropSuite->propSetString(p, kOfxPropLabel, 0, label);
  gPropSuite->propSetInt(p, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
}

static void defChoice(OfxParamSetHandle ps, const char *name, const char *label,
                      int def, const char **options, int numOptions,
                      const char *hint, const char *grp) {
  OfxPropertySetHandle p;
  gParamSuite->paramDefine(ps, kOfxParamTypeChoice, name, &p);
  gPropSuite->propSetString(p, kOfxPropLabel, 0, label);
  gPropSuite->propSetInt(p, kOfxParamPropDefault, 0, def);
  for (int i = 0; i < numOptions; i++)
    gPropSuite->propSetString(p, kOfxParamPropChoiceOption, i, options[i]);
  gPropSuite->propSetInt(p, kOfxParamPropAnimates, 0, 0);
  if (hint)
    gPropSuite->propSetString(p, kOfxParamPropHint, 0, hint);
  if (grp)
    gPropSuite->propSetString(p, kOfxParamPropParent, 0, grp);
}

// ===== ACTIONS =============================================================

static OfxStatus actionLoad() {
  gPropSuite = (OfxPropertySuiteV1 *)gHost->fetchSuite(gHost->host,
                                                       kOfxPropertySuite, 1);
  gParamSuite = (OfxParameterSuiteV1 *)gHost->fetchSuite(gHost->host,
                                                         kOfxParameterSuite, 1);
  gEffSuite = (OfxImageEffectSuiteV1 *)gHost->fetchSuite(
      gHost->host, kOfxImageEffectSuite, 1);
  if (!gPropSuite || !gParamSuite || !gEffSuite)
    return kOfxStatErrMissingHostFeature;
  return kOfxStatOK;
}

static OfxStatus actionDescribe(OfxImageEffectHandle fx) {
  OfxPropertySetHandle ep;
  gEffSuite->getPropertySet(fx, &ep);

  gPropSuite->propSetString(ep, kOfxPropLabel, 0, PLUGIN_NAME);
  gPropSuite->propSetString(ep, kOfxImageEffectPluginPropGrouping, 0,
                            PLUGIN_GROUP);
  gPropSuite->propSetString(ep, kOfxImageEffectPropSupportedContexts, 0,
                            kOfxImageEffectContextFilter);
  gPropSuite->propSetString(ep, kOfxImageEffectPropSupportedPixelDepths, 0,
                            kOfxBitDepthFloat);
  gPropSuite->propSetString(ep, kOfxImageEffectPropMetalRenderSupported, 0,
                            "true");
  gPropSuite->propSetInt(ep, kOfxImageEffectPropTemporalClipAccess, 0, 0);
  gPropSuite->propSetInt(ep, kOfxImageEffectPropSupportsTiles, 0, 1);
  gPropSuite->propSetInt(ep, kOfxImageEffectPropSupportsMultiResolution, 0, 1);
  return kOfxStatOK;
}

static OfxStatus actionDescribeInContext(OfxImageEffectHandle fx) {
  // Clips
  OfxPropertySetHandle cp;
  gEffSuite->clipDefine(fx, kOfxImageEffectSimpleSourceClipName, &cp);
  gPropSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 0,
                            kOfxImageComponentRGBA);

  gEffSuite->clipDefine(fx, kOfxImageEffectOutputClipName, &cp);
  gPropSuite->propSetString(cp, kOfxImageEffectPropSupportedComponents, 0,
                            kOfxImageComponentRGBA);

  // Parameters
  OfxParamSetHandle ps;
  gEffSuite->getParamSet(fx, &ps);

  // Groups
  defGroup(ps, "grpFilm", "Film Stock", true);
  defGroup(ps, "grpGlobal", "Global Settings", true);
  defGroup(ps, "grpLayers", "Grain Architecture (3 Layers)", true);
  defGroup(ps, "grpTexture", "Texture Controls", true);
  defGroup(ps, "grpRGB", "RGB Balance (Arri Style)", false);
  defGroup(ps, "grpResponse", "Film Response Curve", false);
  defGroup(ps, "grpDebug", "Debug", false);

  // Film Stock — 3 dropdown menus
  const char *formatOpts[] = {"8mm", "16mm", "35mm", "70mm / IMAX"};
  defChoice(ps, P_FORMAT, "Format", 2, formatOpts, 4,
            "Physical film format (grain size relative to frame)", "grpFilm");

  const char *stockOpts[] = {
      "Kodak Vision3 250D (5207)", "Kodak Vision3 500T (5219)",
      "Kodak Vision3 200T (5213)", "Fuji Eterna 500T (8673)",
      "Kodak Double-X B&W (5222)"};
  defChoice(ps, P_STOCK, "Emulsion", 0, stockOpts, 5,
            "Film emulsion type (H&D curves, cross-talk, grain response)",
            "grpFilm");

  const char *processOpts[] = {"Classic", "Bleach Bypass", "Reversal"};
  defChoice(ps, P_PROCESS, "Process", 0, processOpts, 3,
            "Photochemical process (grain contrast character)", "grpFilm");

  // Global
  defDouble(ps, P_RESSCALE, "Resolution Scale", 0.7, 0.1, 2.0, 0.01,
            "Fine grain size adjustment", "grpGlobal");
  defDouble(ps, P_SPEED, "Animation Speed", 0.2, 0.0, 1.0, 0.01,
            "Grain temporal speed", "grpGlobal");

  // Layers
  defDouble(ps, P_FINE, "Fine Detail", 0.5, 0.0, 1.0, 0.01, nullptr,
            "grpLayers");
  defDouble(ps, P_MEDIUM, "Medium Body", 0.6, 0.0, 1.0, 0.01, nullptr,
            "grpLayers");
  defDouble(ps, P_COARSE, "Coarse Structure", 0.4, 0.0, 1.0, 0.01, nullptr,
            "grpLayers");

  // Texture
  defDouble(ps, P_AMOUNT, "Global Intensity", 0.4, 0.0, 1.0, 0.01, nullptr,
            "grpTexture");
  defDouble(ps, P_DEPTH, "Grain Depth Contrast", 0.4, 0.0, 1.0, 0.01, nullptr,
            "grpTexture");
  defDouble(ps, P_SOFT, "Grain Softness", 0.6, 0.0, 1.0, 0.01, nullptr,
            "grpTexture");

  // RGB
  defDouble(ps, P_BIAS_R, "Bias Red", 1.2, 0.0, 2.0, 0.01, nullptr, "grpRGB");
  defDouble(ps, P_BIAS_G, "Bias Green", 0.9, 0.0, 2.0, 0.01, nullptr, "grpRGB");
  defDouble(ps, P_BIAS_B, "Bias Blue", 1.1, 0.0, 2.0, 0.01, nullptr, "grpRGB");

  // Response
  defDouble(ps, P_SHADOW, "Shadow Response", 0.5, 0.0, 1.0, 0.01, nullptr,
            "grpResponse");
  defDouble(ps, P_HIGHLIGHT, "Highlight Response", 0.8, 0.0, 1.0, 0.01, nullptr,
            "grpResponse");

  // Debug
  defBool(ps, P_SHOWMASK, "Show Driving Mask", false,
          "Visualise luminance response", "grpDebug");

  return kOfxStatOK;
}

static OfxStatus actionCreateInstance(OfxImageEffectHandle fx) {
  OfxParamSetHandle ps;
  gEffSuite->getParamSet(fx, &ps);
  InstData *d = new InstData;

  gParamSuite->paramGetHandle(ps, P_FORMAT, &d->format, nullptr);
  gParamSuite->paramGetHandle(ps, P_STOCK, &d->stock, nullptr);
  gParamSuite->paramGetHandle(ps, P_PROCESS, &d->process, nullptr);
  gParamSuite->paramGetHandle(ps, P_RESSCALE, &d->resScale, nullptr);
  gParamSuite->paramGetHandle(ps, P_SPEED, &d->speed, nullptr);
  gParamSuite->paramGetHandle(ps, P_FINE, &d->fine, nullptr);
  gParamSuite->paramGetHandle(ps, P_MEDIUM, &d->medium, nullptr);
  gParamSuite->paramGetHandle(ps, P_COARSE, &d->coarse, nullptr);
  gParamSuite->paramGetHandle(ps, P_AMOUNT, &d->amount, nullptr);
  gParamSuite->paramGetHandle(ps, P_DEPTH, &d->depth, nullptr);
  gParamSuite->paramGetHandle(ps, P_SOFT, &d->softness, nullptr);
  gParamSuite->paramGetHandle(ps, P_BIAS_R, &d->biasR, nullptr);
  gParamSuite->paramGetHandle(ps, P_BIAS_G, &d->biasG, nullptr);
  gParamSuite->paramGetHandle(ps, P_BIAS_B, &d->biasB, nullptr);
  gParamSuite->paramGetHandle(ps, P_SHADOW, &d->shadow, nullptr);
  gParamSuite->paramGetHandle(ps, P_HIGHLIGHT, &d->highlight, nullptr);
  gParamSuite->paramGetHandle(ps, P_SHOWMASK, &d->showMask, nullptr);

  OfxPropertySetHandle ep;
  gEffSuite->getPropertySet(fx, &ep);
  gPropSuite->propSetPointer(ep, kOfxPropInstanceData, 0, d);
  return kOfxStatOK;
}

static OfxStatus actionDestroyInstance(OfxImageEffectHandle fx) {
  OfxPropertySetHandle ep;
  gEffSuite->getPropertySet(fx, &ep);
  InstData *d = nullptr;
  gPropSuite->propGetPointer(ep, kOfxPropInstanceData, 0, (void **)&d);
  delete d;
  return kOfxStatOK;
}

// ===== RENDER ==============================================================

static OfxStatus actionRender(OfxImageEffectHandle fx,
                              OfxPropertySetHandle inArgs) {
  // Frame time (deterministic seed)
  double time = 0.0;
  gPropSuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);

  // Instance data
  OfxPropertySetHandle ep;
  gEffSuite->getPropertySet(fx, &ep);
  InstData *d = nullptr;
  gPropSuite->propGetPointer(ep, kOfxPropInstanceData, 0, (void **)&d);

  // Metal command queue
  void *cmdQ = nullptr;
  gPropSuite->propGetPointer(inArgs, kOfxImageEffectPropMetalCommandQueue, 0,
                             &cmdQ);
  if (!cmdQ)
    return kOfxStatErrUnsupported;

  // Clips & images
  OfxImageClipHandle srcClip, dstClip;
  gEffSuite->clipGetHandle(fx, kOfxImageEffectSimpleSourceClipName, &srcClip,
                           nullptr);
  gEffSuite->clipGetHandle(fx, kOfxImageEffectOutputClipName, &dstClip,
                           nullptr);

  OfxPropertySetHandle srcImg = nullptr, dstImg = nullptr;
  gEffSuite->clipGetImage(srcClip, time, nullptr, &srcImg);
  gEffSuite->clipGetImage(dstClip, time, nullptr, &dstImg);
  if (!srcImg || !dstImg) {
    if (srcImg)
      gEffSuite->clipReleaseImage(srcImg);
    if (dstImg)
      gEffSuite->clipReleaseImage(dstImg);
    return kOfxStatFailed;
  }

  // Data pointers & geometry
  void *srcPtr = nullptr;
  void *dstPtr = nullptr;
  gPropSuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
  gPropSuite->propGetPointer(dstImg, kOfxImagePropData, 0, &dstPtr);

  int srcRB = 0, dstRB = 0;
  gPropSuite->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRB);
  gPropSuite->propGetInt(dstImg, kOfxImagePropRowBytes, 0, &dstRB);

  int rw[4]; // render window: x1 y1 x2 y2
  gPropSuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, rw);

  int srcBounds[4];
  gPropSuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, srcBounds);

  // Fill GrainParams using bounds (to process the entire frame DaVinci Resolve
  // allocates)
  GrainParams gp;
  gp.width = srcBounds[2] - srcBounds[0];
  gp.height = srcBounds[3] - srcBounds[1];
  gp.boundsX = srcBounds[0];
  gp.boundsY = srcBounds[1];
  gp.time = (float)time;
  gp.srcRowBytes = srcRB;
  gp.dstRowBytes = dstRB;

  // Read parameter values at current time
  int iv;
  double dv;

  gParamSuite->paramGetValueAtTime(d->format, time, &iv);
  gp.formatSelect = iv;
  gParamSuite->paramGetValueAtTime(d->stock, time, &iv);
  gp.stockSelect = iv;
  gParamSuite->paramGetValueAtTime(d->process, time, &iv);
  gp.processSelect = iv;
  gParamSuite->paramGetValueAtTime(d->resScale, time, &dv);
  gp.resScale = (float)dv;
  gParamSuite->paramGetValueAtTime(d->speed, time, &dv);
  gp.animSpeed = (float)dv;
  gParamSuite->paramGetValueAtTime(d->fine, time, &dv);
  gp.mixFine = (float)dv;
  gParamSuite->paramGetValueAtTime(d->medium, time, &dv);
  gp.mixMedium = (float)dv;
  gParamSuite->paramGetValueAtTime(d->coarse, time, &dv);
  gp.mixCoarse = (float)dv;
  gParamSuite->paramGetValueAtTime(d->amount, time, &dv);
  gp.globalAmt = (float)dv;
  gParamSuite->paramGetValueAtTime(d->depth, time, &dv);
  gp.grainDepth = (float)dv;
  gParamSuite->paramGetValueAtTime(d->softness, time, &dv);
  gp.grainSoftness = (float)dv;
  gParamSuite->paramGetValueAtTime(d->biasR, time, &dv);
  gp.biasR = (float)dv;
  gParamSuite->paramGetValueAtTime(d->biasG, time, &dv);
  gp.biasG = (float)dv;
  gParamSuite->paramGetValueAtTime(d->biasB, time, &dv);
  gp.biasB = (float)dv;
  gParamSuite->paramGetValueAtTime(d->shadow, time, &dv);
  gp.shadowResponse = (float)dv;
  gParamSuite->paramGetValueAtTime(d->highlight, time, &dv);
  gp.highlightResponse = (float)dv;
  gParamSuite->paramGetValueAtTime(d->showMask, time, &iv);
  gp.showMask = iv;

  // Setup color science data specific to the emulsion stock
  static const float toe_data[5][3] = {
      {0.08f, 0.06f, 0.10f}, // 250D
      {0.12f, 0.09f, 0.16f}, // 500T
      {0.10f, 0.07f, 0.13f}, // 200T
      {0.11f, 0.07f, 0.13f}, // Fuji Eterna
      {0.05f, 0.05f, 0.05f}  // B&W
  };
  static const float gamma_data[5][3] = {
      {0.68f, 0.72f, 0.65f}, // 250D
      {0.52f, 0.55f, 0.48f}, // 500T
      {0.50f, 0.53f, 0.47f}, // 200T
      {0.56f, 0.62f, 0.50f}, // Fuji Eterna
      {0.78f, 0.78f, 0.78f}  // B&W
  };
  static const float shoulder_data[5][3] = {
      {0.92f, 0.90f, 0.88f}, // 250D
      {0.85f, 0.84f, 0.80f}, // 500T
      {0.88f, 0.87f, 0.83f}, // 200T
      {0.82f, 0.88f, 0.80f}, // Fuji Eterna
      {0.88f, 0.88f, 0.88f}  // B&W
  };
  static const float crosstalk_data[5][9] = {
      {0.95f, 0.03f, 0.02f, 0.01f, 0.97f, 0.02f, 0.02f, 0.04f, 0.94f}, // 250D
      {0.93f, 0.04f, 0.03f, 0.02f, 0.96f, 0.02f, 0.04f, 0.06f, 0.90f}, // 500T
      {0.94f, 0.04f, 0.02f, 0.01f, 0.97f, 0.02f, 0.03f, 0.05f, 0.92f}, // 200T
      {0.94f, 0.03f, 0.03f, 0.02f, 0.98f, 0.00f, 0.03f, 0.05f,
       0.92f}, // Fuji Eterna
      {0.2126f, 0.7152f, 0.0722f, 0.2126f, 0.7152f, 0.0722f, 0.2126f, 0.7152f,
       0.0722f} // B&W
  };
  static const float grainSize_data[5][3] = {
      {0.45f, 0.40f, 0.55f}, // 250D
      {0.70f, 0.62f, 0.80f}, // 500T
      {0.38f, 0.33f, 0.45f}, // 200T
      {0.65f, 0.60f, 0.72f}, // Fuji Eterna
      {1.10f, 1.10f, 1.10f}  // B&W
  };
  static const float grainCorr_data[5][2] = {
      {0.15f, 0.10f}, // 250D
      {0.12f, 0.08f}, // 500T
      {0.18f, 0.12f}, // 200T
      {0.15f, 0.10f}, // Fuji Eterna
      {0.30f, 0.30f}  // B&W
  };

  int s = gp.stockSelect;
  if (s < 0 || s > 4)
    s = 0;

  for (int i = 0; i < 3; ++i) {
    gp.toe[i] = toe_data[s][i];
    gp.gamma[i] = gamma_data[s][i];
    gp.shoulder[i] = shoulder_data[s][i];
    gp.grainSize[i] = grainSize_data[s][i];
  }
  for (int i = 0; i < 9; ++i) {
    gp.crosstalk[i] = crosstalk_data[s][i];
  }
  gp.grainCorr[0] = grainCorr_data[s][0];
  gp.grainCorr[1] = grainCorr_data[s][1];

  // --- Precalculate Performance Parameters (PERF-2) ---
  float format_scale = 1.0f;
  if (gp.formatSelect == 0) format_scale = 4.0f;
  else if (gp.formatSelect == 1) format_scale = 2.0f;
  else if (gp.formatSelect == 2) format_scale = 1.0f;
  else if (gp.formatSelect == 3) format_scale = 0.4f;
  gp.format_scale = format_scale;

  float contrast_mod = 1.0f;
  if (gp.processSelect == 1) contrast_mod = 1.5f;
  else if (gp.processSelect == 2) contrast_mod = 1.2f;
  gp.contrast_mod = contrast_mod;

  gp.bw_mode = (gp.stockSelect == 4) ? 1.0f : 0.0f;

  // Dispatch Metal
  int result = RunMetalKernel(cmdQ, srcPtr, dstPtr, &gp);

  gEffSuite->clipReleaseImage(srcImg);
  gEffSuite->clipReleaseImage(dstImg);

  return (result == 0) ? kOfxStatOK : kOfxStatFailed;
}

// ===== MAIN DISPATCHER =====================================================

static OfxStatus pluginMain(const char *action, const void *handle,
                            OfxPropertySetHandle inArgs,
                            OfxPropertySetHandle outArgs) {
  auto fx = (OfxImageEffectHandle)handle;

  if (!strcmp(action, kOfxActionLoad))
    return actionLoad();
  if (!strcmp(action, kOfxActionUnload))
    return kOfxStatOK;
  if (!strcmp(action, kOfxActionDescribe))
    return actionDescribe(fx);
  if (!strcmp(action, kOfxImageEffectActionDescribeInContext))
    return actionDescribeInContext(fx);
  if (!strcmp(action, kOfxActionCreateInstance))
    return actionCreateInstance(fx);
  if (!strcmp(action, kOfxActionDestroyInstance))
    return actionDestroyInstance(fx);
  if (!strcmp(action, kOfxImageEffectActionRender))
    return actionRender(fx, inArgs);

  return kOfxStatReplyDefault;
}

// ===== ENTRY POINTS ========================================================

static void setHost(OfxHost *h) { gHost = h; }

static OfxPlugin gPlugin = {kOfxImageEffectPluginApi,
                            1, // API version
                            PLUGIN_ID,
                            PLUGIN_VMAJ,
                            PLUGIN_VMIN,
                            setHost,
                            pluginMain};

OfxExport int OfxGetNumberOfPlugins(void) { return 1; }
OfxExport OfxPlugin *OfxGetPlugin(int /*nth*/) { return &gPlugin; }
