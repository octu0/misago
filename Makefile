SCHEME_PKG := misago-Package
SCHEME_TRAIN := train
SCHEME_TRAIN_DIGITS := train-digits
DESTINATION := platform=macOS
CONFIG := Release
# xcodebuild output goes under .build so the products have a known path (the SwiftPM-generated schemes do not
# report BUILT_PRODUCTS_DIR through -showBuildSettings).
DERIVED := .build/xcode
XCODEBUILD := xcodebuild -destination '$(DESTINATION)' -configuration $(CONFIG) -derivedDataPath $(DERIVED)
PRODUCTS := $(DERIVED)/Build/Products/$(CONFIG)

TRAIN ?= .tmp/datasets/train.msgo
EVAL ?= .tmp/datasets/eval.msgo
EPOCHS ?= 20
WEIGHTS ?= .tmp/weights/model.json
DIGIT_TRAIN ?= .tmp/datasets/digits_train.msgd
DIGIT_EVAL ?= .tmp/datasets/digits_eval.msgd
DIGIT_EPOCHS ?= 40
DIGIT_WEIGHTS ?= .tmp/weights/digits.json
ARGS ?=

IMAGE ?= misago-api

.PHONY: all build build-train build-train-digits test train eval train-digits prepare prepare-digits label synth classify read benchmark api docker clean help

all: build test

# MLX's Metal shaders are only built by Xcode, so everything that touches MLX goes through xcodebuild.
build:
	$(XCODEBUILD) build -scheme $(SCHEME_PKG)

build-train:
	$(XCODEBUILD) build -scheme $(SCHEME_TRAIN)

test:
	$(XCODEBUILD) test -scheme $(SCHEME_PKG) ENABLE_TESTABILITY=YES

train: build-train
	$(PRODUCTS)/train --train $(TRAIN) --eval $(EVAL) -e $(EPOCHS) --export-weights $(WEIGHTS) $(ARGS)

eval: build-train
	$(PRODUCTS)/train --train $(TRAIN) --eval $(EVAL) -e 0 --import-weights $(WEIGHTS) $(ARGS)

build-train-digits:
	$(XCODEBUILD) build -scheme $(SCHEME_TRAIN_DIGITS)

# DIGIT_TRAIN may list several datasets (synthetic and real); each becomes a --train argument.
train-digits: build-train-digits
	$(PRODUCTS)/train-digits $(foreach f,$(DIGIT_TRAIN),--train $(f)) --eval $(DIGIT_EVAL) -e $(DIGIT_EPOCHS) --export-weights $(DIGIT_WEIGHTS) $(ARGS)

# The tools below do not touch MLX and run through SwiftPM (Release only; debug builds lose SIMD).
prepare:
	swift run -c release prepare $(ARGS)

prepare-digits:
	swift run -c release prepare-digits $(ARGS)

# macOS only: labels number fields with the system OCR for review.
label:
	swift run -c release label $(ARGS)

# macOS only: renders synthetic number fields with CoreText.
synth:
	swift run -c release synth $(ARGS)

classify:
	swift run -c release classify $(ARGS)

read:
	swift run -c release read $(ARGS)

benchmark:
	swift run -c release benchmark $(ARGS)

# HTTP API (Vapor). Models are read from .tmp/models (see MISAGO_* in Sources/MisagoAPI), the page from .tmp/public.
api:
	swift run -c release misago-api serve --env development $(ARGS)

# Linux container for Cloud Run (needs .tmp/models and .tmp/public staged first).
docker:
	docker build -t $(IMAGE) .

clean:
	$(XCODEBUILD) clean -scheme $(SCHEME_PKG)
	swift package clean

help:
	@echo "make build        Build every target (Release, xcodebuild)"
	@echo "make build-train  Build the train tool (Release)"
	@echo "make test         Run the test suite (Release, xcodebuild)"
	@echo "make train TRAIN=<train.msgo> EVAL=<eval.msgo> [EPOCHS=20 WEIGHTS=... ARGS=...]"
	@echo "make eval  TRAIN=<train.msgo> EVAL=<eval.msgo> WEIGHTS=<model.json>"
	@echo "make prepare ARGS=\"--class <name>=<dir> ... --train <out.msgo> --eval <out.msgo>\""
	@echo "make label ARGS=\"--regions <regions.json> --class <name>=<dir> --out <labels.jsonl> --sheets <dir>\""
	@echo "make synth ARGS=\"--font <ttf> --background <dir> --count <N> --out <out.msgd>\""
	@echo "make prepare-digits ARGS=\"--labels <labels.jsonl> --train <out.msgd> --eval <out.msgd>\""
	@echo "make train-digits DIGIT_TRAIN=\"<synth.msgd> <real.msgd>\" DIGIT_EVAL=<eval.msgd> [DIGIT_EPOCHS=40 DIGIT_WEIGHTS=...]"
	@echo "make classify ARGS=\"--weights <model.json> <image>...\""
	@echo "make read ARGS=\"--weights <model.json> --digit-weights <digits.json> --regions <regions.json> <image>...\""
	@echo "make benchmark ARGS=\"[--weights <model.json>]\""
	@echo "make api          Run the HTTP API locally (models in .tmp/models, page in .tmp/public)"
	@echo "make docker       Build the Cloud Run image (IMAGE=misago-api)"
	@echo "make clean        Remove build products"
