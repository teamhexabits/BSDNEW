GENCCODE_ASSEMBLY_TYPE=-a gcc-darwin
SO=dylib
SOBJ=dylib
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.78.3.dylib
COMPILE=arm64-apple-darwin20.0.0-clang -DU_ALL_IMPLEMENTATION -DU_ATTRIBUTE_DEPRECATED= -D_FORTIFY_SOURCE=2 -isystem /Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/include -mmacosx-version-min=11.0 -mmacosx-version-min=11.0    -ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -isystem /Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/include -fdebug-prefix-map=/Users/runner/miniforge3/conda-bld/bld/rattler-build_icu_1786545847/work=/usr/local/src/conda/icu-78.3 -fdebug-prefix-map=/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -fno-common -c
LIBFLAGS=-I/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/include  -dynamic
GENLIB=arm64-apple-darwin20.0.0-clang -dynamiclib -dynamic -ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -isystem /Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/include -fdebug-prefix-map=/Users/runner/miniforge3/conda-bld/bld/rattler-build_icu_1786545847/work=/usr/local/src/conda/icu-78.3 -fdebug-prefix-map=/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/lib -L/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/lib -Wl,-rpath,/Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/lib 
LDICUDTFLAGS=
LD_SONAME=-Wl,-compatibility_version -Wl,78 -Wl,-current_version -Wl,78.3 -install_name /Users/tusharmaheshkhandade/Downloads/border-doc-screen/.tools/tesseract/lib/
RPATH_FLAGS=
BIR_LDFLAGS=
AR=arm64-apple-darwin20.0.0-ar
ARFLAGS=r -c
RANLIB=arm64-apple-darwin20.0.0-ranlib
INSTALL_CMD=/usr/bin/install -c
