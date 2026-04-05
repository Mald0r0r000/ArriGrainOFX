# ArriGrain OFX Plugin

A physically-accurate film grain emulation plugin for DaVinci Resolve, built with Metal GPU acceleration.

## Features

- **Density-driven grain amplitude** — Grain intensity varies per-channel based on pixel density (log space)
- **Multi-octave noise** — Fine, medium, and coarse grain layers with independent temporal coherence
- **Clumping multiplicative** — Coarse grain acts as envelope for realistic halide crystal clustering
- **Cross-channel correlation** — Configurable R-G and G-B grain correlation per format
- **Resolution independence** — Grain size normalized to image diagonal for consistent appearance
- **Format presets** — 8mm, 16mm, 35mm, 70mm/IMAX with appropriate grain characteristics

## Requirements

- macOS 12.0+ (Apple Silicon or Intel)
- DaVinci Resolve 17+ (Studio or Free)

## Installation

### Option 1: Download Release (Recommended)

1. Download the latest release ZIP from the [Releases](https://github.com/Mald0r0r000/ArriGrainOFX/releases) page
2. Unzip the downloaded file
3. Open Terminal and navigate to the unzipped folder:
   ```bash
   cd /path/to/unzipped/folder
   ```
4. Make the installer executable and run it:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```
5. Enter your Mac password when prompted
6. Restart DaVinci Resolve

### Option 2: Manual Installation

1. Download the latest release ZIP
2. Unzip and copy `ArriGrainOFX.ofx.bundle` to `/Library/OFX/Plugins/`
3. Open Terminal and run:
   ```bash
   sudo xattr -rd com.apple.quarantine /Library/OFX/Plugins/ArriGrainOFX.ofx.bundle
   sudo codesign -f -s - /Library/OFX/Plugins/ArriGrainOFX.ofx.bundle
   ```
4. Restart DaVinci Resolve

### Option 3: Build from Source

```bash
# Clone the repository
git clone https://github.com/Mald0r0r000/ArriGrainOFX.git
cd ArriGrainOFX

# Build and install
make clean && make && sudo make install
```

## Troubleshooting

### "Plugin not found" in DaVinci Resolve

1. Ensure the plugin is installed in `/Library/OFX/Plugins/`
2. Run the quarantine removal command:
   ```bash
   sudo xattr -rd com.apple.quarantine /Library/OFX/Plugins/ArriGrainOFX.ofx.bundle
   sudo codesign -f -s - /Library/OFX/Plugins/ArriGrainOFX.ofx.bundle
   ```
3. Restart DaVinci Resolve

### "Unidentified developer" warning

This is normal for unsigned plugins. The `install.sh` script handles this by applying an ad-hoc signature. If installing manually, run:
```bash
sudo codesign -f -s - /Library/OFX/Plugins/ArriGrainOFX.ofx.bundle
```

### Plugin crashes or doesn't render

1. Check Console.app for "ArriGrain" messages
2. Ensure you're using macOS 12.0 or later
3. Try the "Test Pattern (Red)" debug mode to verify the kernel is running

## Parameters

| Parameter | Description |
|-----------|-------------|
| **Format** | Film format preset (8mm, 16mm, 35mm, 70mm) |
| **Process** | Development process (Normal, Push, Pull) |
| **Resolution Scale** | Adjust grain size independently |
| **Animation Speed** | Temporal grain evolution speed |
| **Fine/Medium/Coarse Mix** | Balance between grain octaves |
| **Global Amount** | Overall grain intensity |
| **Grain Depth** | Grain contrast/intensity |
| **Grain Softness** | Fine grain blur amount |
| **RGB Bias** | Per-channel grain intensity |
| **Shadow Response** | Grain behavior in shadows |
| **Highlight Response** | Grain behavior in highlights |

## Debug Modes

| Mode | Description |
|------|-------------|
| Off | Normal grain rendering |
| Density Mask | Shows per-channel grain amplitude |
| Grain Only | Raw noise pattern |
| Amplitude R | Red channel amplitude as grayscale |
| Pass Through | Input unchanged (verify kernel runs) |
| Test Pattern (Red) | Solid red (verify kernel compiles) |

## Version History

### v2.0
- Density-driven grain amplitude per channel
- Clumping multiplicative (coarse as envelope)
- Cross-channel correlation
- Resolution independence
- Fixed grain size varying by format
- Removed unused emulsion type parameter

### v1.0
- Initial release with simplex noise grain generation
- Multi-octave grain layers
- Soft light blend mode

## License

MIT License - See LICENSE file for details.

## Credits

- OpenSimplex2S noise algorithm
- Metal GPU compute framework
- OFX plugin architecture
