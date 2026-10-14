// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "misago",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Misago", targets: ["Misago"]),
        .library(name: "MisagoTraining", targets: ["MisagoTraining"]),
        .executable(name: "train", targets: ["train"]),
        .executable(name: "prepare", targets: ["prepare"]),
        .executable(name: "classify", targets: ["classify"]),
        .executable(name: "benchmark", targets: ["benchmark"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
    ],
    targets: [
        // Inference library: Foundation and Accelerate only, no MLX.
        .target(
            name: "Misago",
            path: "Sources/Misago"
        ),
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
        // Image files -> normalized screen images -> dataset file (ImageIO is used here and in classify only).
        .executableTarget(name: "prepare", dependencies: ["Misago"], path: "script/prepare"),
        // One image -> verdict.
        .executableTarget(name: "classify", dependencies: ["Misago"], path: "script/classify"),
        // CPU inference speed.
        .executableTarget(name: "benchmark", dependencies: ["Misago"], path: "script/benchmark"),
        .testTarget(
            name: "MisagoTests",
            dependencies: ["Misago", "MisagoTraining", .product(name: "MLX", package: "mlx-swift")],
            path: "Tests/MisagoTests"
        )
    ]
)
