// swift-tools-version: 6.0
import PackageDescription

// The inference library, the HTTP API and the image decoder build everywhere (macOS and Linux).
// Training (MLX, Metal) and the image tools that use ImageIO, Vision or CoreText are macOS only.
var products: [Product] = [
    .library(name: "Misago", targets: ["Misago"]),
    .library(name: "ImageDecoding", targets: ["ImageDecoding"]),
    .executable(name: "misago-api", targets: ["MisagoAPI"]),
    .executable(name: "benchmark", targets: ["benchmark"])
]
var dependencies: [Package.Dependency] = [
    .package(url: "https://github.com/vapor/vapor.git", from: "4.99.0")
]
var targets: [Target] = [
    // Inference library: Foundation only, no Apple frameworks.
    .target(name: "Misago", path: "Sources/Misago"),
    // stb_image (public domain) for decoding uploads without ImageIO.
    .target(name: "CStbImage", path: "Sources/CStbImage"),
    .target(name: "ImageDecoding", dependencies: ["CStbImage"], path: "Sources/ImageDecoding"),
    // HTTP API and static page server (Vapor).
    .executableTarget(
        name: "MisagoAPI",
        dependencies: ["Misago", "ImageDecoding", .product(name: "Vapor", package: "vapor")],
        path: "Sources/MisagoAPI"
    ),
    // CPU inference speed.
    .executableTarget(name: "benchmark", dependencies: ["Misago"], path: "script/benchmark"),
    .testTarget(name: "ImageDecodingTests", dependencies: ["ImageDecoding"], path: "Tests/ImageDecodingTests")
]

#if os(macOS)
products += [
    .library(name: "MisagoTraining", targets: ["MisagoTraining"]),
    .executable(name: "train", targets: ["train"]),
    .executable(name: "train-digits", targets: ["train-digits"]),
    .executable(name: "prepare", targets: ["prepare"]),
    .executable(name: "prepare-digits", targets: ["prepare-digits"]),
    .executable(name: "classify", targets: ["classify"]),
    .executable(name: "read", targets: ["read"]),
    .executable(name: "label", targets: ["label"]),
    .executable(name: "synth", targets: ["synth"])
]
dependencies += [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
]
targets += [
    // Training library: MLX (Metal GPU).
    .target(
        name: "MisagoTraining",
        dependencies: [
            "Misago",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift")
        ],
        path: "Sources/MisagoTraining"
    ),
    // Dataset file -> MLX training -> weights JSON -> held-out evaluation and calibration.
    .executableTarget(
        name: "train",
        dependencies: ["Misago", "MisagoTraining", .product(name: "MLX", package: "mlx-swift")],
        path: "script/train"
    ),
    // Digit-strip dataset -> CTC training -> weights JSON.
    .executableTarget(
        name: "train-digits",
        dependencies: ["Misago", "MisagoTraining", .product(name: "MLX", package: "mlx-swift")],
        path: "script/train-digits"
    ),
    // Image files -> normalized screen images -> dataset file (ImageIO).
    .executableTarget(name: "prepare", dependencies: ["Misago"], path: "script/prepare"),
    // Labels JSONL -> digit-strip dataset (ImageIO).
    .executableTarget(name: "prepare-digits", dependencies: ["Misago"], path: "script/prepare-digits"),
    // One image -> verdict (ImageIO).
    .executableTarget(name: "classify", dependencies: ["Misago"], path: "script/classify"),
    // Whole pipeline on one image: class and number (ImageIO).
    .executableTarget(name: "read", dependencies: ["Misago"], path: "script/read"),
    // Number-field labeling with the system OCR (Vision), for review before training the reader.
    .executableTarget(name: "label", dependencies: ["Misago"], path: "script/label"),
    // Synthetic number fields (CoreText) for pretraining the digit reader.
    .executableTarget(name: "synth", dependencies: ["Misago"], path: "script/synth"),
    .testTarget(
        name: "MisagoTests",
        dependencies: ["Misago", "MisagoTraining", .product(name: "MLX", package: "mlx-swift")],
        path: "Tests/MisagoTests"
    )
]
#endif

let package = Package(
    name: "misago",
    platforms: [
        .macOS(.v14)
    ],
    products: products,
    dependencies: dependencies,
    targets: targets
)
