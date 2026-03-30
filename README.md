# ArriGrainOFX

A film grain OFX plugin with authentic Arriflex-style grain simulation using Metal GPU acceleration.

## Features

- **Multi-scale grain**: Fine, medium, and coarse grain layers with independent mixing
- **Film response curves**: H&D (Hurter-Driffield) curves for authentic film response
- **Cross-talk**: RGB channel correlation for realistic color grain behavior
- **Luminance masking**: Grain intensity adapts to image luminance (more grain in shadows, less in highlights)
- **Real-time animation**: Temporal grain evolution with configurable speed

## Requirements

- macOS 10.15+ (Metal support)
- Xcode Command Line Tools
- OFX-compatible host (DaVinci Resolve, etc.)

## Building

```bash
make clean && make
```

The compiled plugin is output as `ArriGrainOFX.ofx.bundle/`.

## Installation

Copy `ArriGrainOFX.ofx.bundle` to your OFX plugins directory:
- DaVinci Resolve: `/Library/OFX/Plugins/`

## Changelog

### v1.1.0 — Bugfix: Grain Visibility

**Fixed three critical issues causing grain to be nearly invisible:**

1. **SIMPLEX_3TAP normalization** — Removed erroneous `* 0.5f` factor that was crushing the noise signal from [-1,1] to [-0.166,0.166]. The macro now correctly averages 3 simplex taps with `* 0.333333f` only.

2. **Layer mixing weight normalization** — Added normalization when mixing fine/medium/coarse grain layers. Previously, weights could sum to >1.0 (default: 0.5+0.6+0.4=1.5), causing unpredictable amplitude scaling. Now divides by weight sum for consistent output.

3. **Grain mask floor** — Raised `floor_val` from `0.12f * shadow_r` to `0.25f * shadow_r` to ensure grain remains visible across all luminance values, especially in midtones where the bell curve was too aggressive.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `globalAmt` | 0.4 | Overall grain intensity |
| `grainDepth` | 0.4 | Grain contrast/depth |
| `grainSoftness` | 1.0 | Spatial blur of grain |
| `mixFine` | 0.5 | Fine grain layer weight |
| `mixMedium` | 0.6 | Medium grain layer weight |
| `mixCoarse` | 0.4 | Coarse grain layer weight |
| `biasR/G/B` | 1.0 | Per-channel grain intensity |
| `shadowResponse` | 1.0 | Grain in shadows |
| `highlightResponse` | 0.8 | Grain suppression in highlights |
| `animSpeed` | 1.0 | Temporal grain evolution speed |
| `showMask` | 0 | Debug: show luminance mask only |

## License

Proprietary — All rights reserved.
