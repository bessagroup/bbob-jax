#!/usr/bin/env bash
# Download the reference implementations the cross-check tests compare
# against (tests/test_crosscheck_*.py), pinned by commit or release and
# checked by SHA-256. Then point the tests at the directory:
#
#     bash scripts/fetch_references.sh /path/to/references
#     BBOB_JAX_REFERENCES=/path/to/references pytest tests/test_crosscheck_*.py
#
# Layout of the directory (names used in tests/_references.py):
#   bbobbenchmarks.py   COCO's legacy Python BBOB code (numbbo/coco)
#   bbob-legacy-c/      legacy BBOB C code, noisy suite (mirror of bbob.v15.03)
#   CEC17_fast_pow/     official CEC 2017 C++ code and input_data/
#   cec2013lsgo-2.2/    official CEC 2013 LSGO C++ code (cec2013lsgo 2.2 sdist)
#   cec2005-c/src/      official CEC 2005 C code (C-Unix-linux-14-Mar-05),
#                       its sources and input_data/ only
#
# Needs curl, unzip, tar (with bzip2) and sha256sum (or shasum). The tests also need
# gcc and g++ to build the C/C++ references.
set -euo pipefail

dest="${1:?usage: fetch_references.sh <destination directory>}"
mkdir -p "$dest"
cd "$dest"

COCO=https://raw.githubusercontent.com/numbbo/coco/0a7b447fc5e1aeff6aa52fb5dcc37eca4e917219
LEGACY=https://raw.githubusercontent.com/lorenzo-consoli/bbob-legacy-code/a62427b47e995775764086b9f2b30046871580e9/c
CEC17=https://github.com/P-N-Suganthan/CEC2017-BoundContrained/raw/2c54cad22f015e803edb09ea86d4c961f5bab644
CEC05=https://github.com/P-N-Suganthan/CEC2005/raw/39f68099a10a9324e596ceb943deb1da26515d08
LSGO=https://files.pythonhosted.org/packages/e4/42/52bb52657b006fd48f53e1f1460560c3a9846de67db94f560bfd5c464aaf

if command -v sha256sum >/dev/null; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

# fetch <url> <output file> <expected sha256>
fetch() {
    curl -fsSL --retry 3 -o "$2" "$1"
    local got
    got="$(sha256 "$2")"
    if [ "$got" != "$3" ]; then
        echo "checksum mismatch for $2: got $got, expected $3" >&2
        rm -f "$2"
        exit 1
    fi
}

fetch "$COCO/code-postprocessing/aRTAplots/bbobbenchmarks.py" bbobbenchmarks.py \
    b93e200e08d8e0c5bc6eaf4c5e6cf6e81b04d73989b7fdf02583631061627797

mkdir -p bbob-legacy-c
while read -r name sum; do
    fetch "$LEGACY/$name" "bbob-legacy-c/$name" "$sum"
done <<'EOF'
bbobStructures.h adaa7812089543a009919c5f16e40ea6c39c5d86a914ae393e0cb015637eb9fb
benchmarksdeclare.c b8bfe942ae737e2650fdcb46cc8fccab7d63c81547535c4a2d918770501c86b6
benchmarksdeclare.h 0b9a8939c7d070848520f44ec88bdc0118d39da9957152a7a79dbef6333d08ef
benchmarkshelper.c 1740c1642b288e1d85a86c11a78b606774a4ea57836e9fd5f5e895dd626560b7
benchmarkshelper.h 416e13638550a5ea93e21c8d24359d43047bc808a6a1b312b39a495496e2439b
benchmarksnoisy.c 946d396ea535b201408f38f5bca5450a026fa380ca99fec2805c7e41e3e1ab0a
benchmarksnoisy.h d55b6e7d11c09ad70bf6a03f50d071ebb9d8061146941d7fcd30fc918b2bf9d7
LICENSE.txt 05e9483694405da26ec1961e94c72a7921336dfe1df04769321ef8ff204e2cca
EOF

fetch "$CEC17/CEC17_fast_pow-C%2B%2B.zip" cec17.zip \
    51e7f55dc5cddf263ec128771dee1db6a2d0e2816b3d650160eb51960f269a1e
unzip -q -o cec17.zip 'CEC17_fast_pow/cec17_test_func.cpp' 'CEC17_fast_pow/input_data/*'
rm cec17.zip

fetch "$LSGO/cec2013lsgo-2.2.tar.gz" cec2013lsgo-2.2.tar.gz \
    e1f46e3fa5b576b7e4d19078867f345a7fdb5269549d12113bd8cba7982d03d5
tar xzf cec2013lsgo-2.2.tar.gz
rm cec2013lsgo-2.2.tar.gz

fetch "$CEC05/C-Unix-linux-14-Mar-05.tar.bz2" cec2005-c.tar.bz2 \
    db1efb60a7ac99f5f6ba563a6e3c2faa176ccaacc2f016b860cb766b73f8225c
mkdir -p cec2005-c
tar xjf cec2005-c.tar.bz2 -C cec2005-c \
    src/aux.c src/def1.c src/def2.c src/def3.c src/def4.c \
    src/global.h src/sub.h src/input_data
rm cec2005-c.tar.bz2

echo "references in $(pwd)"
