# Makefile — Arri Style Grain OFX Plugin (Metal, Apple Silicon)
#
# The Metal kernel source is embedded in MetalKernel.mm and compiled
# at runtime by the Metal framework. No offline Metal compiler needed.
#
# Usage:
#   ./setup.sh             # Download OFX SDK headers (once)
#   make                   # Build plugin bundle
#   make install           # Copy to /Library/OFX/Plugins/
#   make clean             # Remove build artifacts

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
PLUGIN_NAME    := ArriGrainOFX
BUNDLE         := $(PLUGIN_NAME).ofx.bundle
OFX_INC        := openfx/include

CXX            := clang++
OBJCXX         := clang++

ARCH           := arm64
MIN_MACOS      := 12.0

CXXFLAGS       := -std=c++17 -arch $(ARCH) -mmacosx-version-min=$(MIN_MACOS) \
                  -O2 -fPIC -fvisibility=hidden \
                  -I$(OFX_INC) -I.

OBJCXXFLAGS    := $(CXXFLAGS) -fobjc-arc

LDFLAGS        := -dynamiclib -arch $(ARCH) -mmacosx-version-min=$(MIN_MACOS) \
                  -framework Metal -framework Foundation \
                  -fvisibility=hidden

INSTALL_DIR    := /Library/OFX/Plugins

# ----------------------------------------------------------------
# Source files
# ----------------------------------------------------------------
PLUGIN_OBJ     := ArriGrainPlugin.o
METAL_BRIDGE   := MetalKernel.o
OFX_DYLIB      := $(PLUGIN_NAME).ofx

# ----------------------------------------------------------------
# Targets
# ----------------------------------------------------------------
.PHONY: all bundle install clean check

all: bundle

# Check OFX headers exist
$(OFX_INC)/ofxCore.h:
	@echo "❌ OFX SDK headers not found. Run: ./setup.sh"
	@exit 1

# Compile C++ plugin
$(PLUGIN_OBJ): ArriGrainPlugin.cpp MetalKernel.h $(OFX_INC)/ofxCore.h
	@echo "🔧 Compiling OFX plugin..."
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile Objective-C++ Metal bridge (includes embedded MSL source)
$(METAL_BRIDGE): MetalKernel.mm MetalKernel.h
	@echo "🔧 Compiling Metal bridge..."
	$(OBJCXX) $(OBJCXXFLAGS) -c -o $@ $<

# Link into .ofx dylib
$(OFX_DYLIB): $(PLUGIN_OBJ) $(METAL_BRIDGE)
	@echo "🔗 Linking $(OFX_DYLIB)..."
	$(CXX) $(LDFLAGS) -o $@ $^

# Assemble .ofx.bundle
bundle: $(OFX_DYLIB) Info.plist
	@echo "📦 Creating bundle..."
	mkdir -p $(BUNDLE)/Contents/MacOS
	cp $(OFX_DYLIB)  $(BUNDLE)/Contents/MacOS/
	cp Info.plist     $(BUNDLE)/Contents/
	@echo "✅ Bundle ready: $(BUNDLE)"

# Install to system OFX directory
install: bundle
	@echo "📂 Installing to $(INSTALL_DIR)/..."
	sudo mkdir -p $(INSTALL_DIR)
	sudo cp -R $(BUNDLE) $(INSTALL_DIR)/
	@echo "✅ Installed. Restart DaVinci Resolve to load the plugin."

# Verify build
check: bundle
	@echo "--- Bundle structure ---"
	@find $(BUNDLE) -type f
	@echo "--- OFX symbols ---"
	@nm -gU $(BUNDLE)/Contents/MacOS/$(OFX_DYLIB) | grep -i ofx || true

# Clean
clean:
	rm -f $(PLUGIN_OBJ) $(METAL_BRIDGE) $(OFX_DYLIB)
	rm -rf $(BUNDLE)
	@echo "🧹 Clean"
