SCHEME_PKG := misago-Package
SCHEME_TRAIN := train
DESTINATION := platform=macOS
CONFIG := Release
PRODUCTS = $$(xcodebuild -scheme $(SCHEME_TRAIN) -destination '$(DESTINATION)' -configuration $(CONFIG) -showBuildSettings 2>/dev/null | awk '/ BUILT_PRODUCTS_DIR =/{print $$3; exit}')

TRAIN ?= .tmp/datasets/train.msgo
EVAL ?= .tmp/datasets/eval.msgo
EPOCHS ?= 20
WEIGHTS ?= .tmp/weights/model.json
ARGS ?=

.PHONY: all build build-train test train eval prepare classify benchmark clean help

all: build test

# MLX's Metal shaders are only built by Xcode, so everything that touches MLX goes through xcodebuild.
build:
	xcodebuild build -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)

build-train:
	xcodebuild build -scheme $(SCHEME_TRAIN) -destination '$(DESTINATION)' -configuration $(CONFIG)

test:
	xcodebuild test -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG) ENABLE_TESTABILITY=YES

train: build-train
	$(PRODUCTS)/train --train $(TRAIN) --eval $(EVAL) -e $(EPOCHS) --export-weights $(WEIGHTS) $(ARGS)

eval: build-train
	$(PRODUCTS)/train --train $(TRAIN) --eval $(EVAL) -e 0 --import-weights $(WEIGHTS) $(ARGS)

# The tools below do not touch MLX and run through SwiftPM (Release only; debug builds lose SIMD).
prepare:
	swift run -c release prepare $(ARGS)

classify:
	swift run -c release classify $(ARGS)

benchmark:
	swift run -c release benchmark $(ARGS)

clean:
	xcodebuild clean -scheme $(SCHEME_PKG) -destination '$(DESTINATION)' -configuration $(CONFIG)
	swift package clean

help:
	@echo "make build        Build every target (Release, xcodebuild)"
	@echo "make build-train  Build the train tool (Release)"
	@echo "make test         Run the test suite (Release, xcodebuild)"
	@echo "make train TRAIN=<train.msgo> EVAL=<eval.msgo> [EPOCHS=20 WEIGHTS=... ARGS=...]"
	@echo "make eval  TRAIN=<train.msgo> EVAL=<eval.msgo> WEIGHTS=<model.json>"
	@echo "make prepare ARGS=\"--positive <dir> --negative <dir> --train <out> --eval <out>\""
	@echo "make classify ARGS=\"--weights <model.json> <image>...\""
	@echo "make benchmark ARGS=\"[--weights <model.json>]\""
	@echo "make clean        Remove build products"
