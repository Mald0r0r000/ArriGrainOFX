#!/bin/bash
# setup.sh — Download OpenFX SDK headers
# Run once before building: ./setup.sh

set -e

OFX_DIR="openfx/include"

if [ -f "$OFX_DIR/ofxCore.h" ]; then
    echo "✅ OFX SDK headers already present in $OFX_DIR"
    exit 0
fi

echo "📥 Downloading OpenFX SDK headers..."
mkdir -p "$OFX_DIR"

REPO_URL="https://raw.githubusercontent.com/AcademySoftwareFoundation/openfx/main/include"

HEADERS=(
    "ofxCore.h"
    "ofxDialog.h"
    "ofxImageEffect.h"
    "ofxInteract.h"
    "ofxKeySyms.h"
    "ofxMemory.h"
    "ofxMessage.h"
    "ofxMultiThread.h"
    "ofxNatron.h"
    "ofxOld.h"
    "ofxOpenGLRender.h"
    "ofxParam.h"
    "ofxParametricParam.h"
    "ofxPixels.h"
    "ofxProgress.h"
    "ofxProperty.h"
    "ofxSonyVegas.h"
    "ofxTimeLine.h"
)

for h in "${HEADERS[@]}"; do
    echo "  ↓ $h"
    curl -sL "$REPO_URL/$h" -o "$OFX_DIR/$h"
done

echo "✅ OFX SDK headers downloaded to $OFX_DIR"
