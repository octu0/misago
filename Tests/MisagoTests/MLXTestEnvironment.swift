import Foundation
import XCTest

/// Checks whether MLX's Metal library is available before any MLX call.
///
/// mlx-swift's Metal shaders are only built by Xcode. A `swift test` binary built by SwiftPM has no metallib and
/// crashes with "Failed to load the default metallib" on the first MLX call, so MLX tests are skipped there and
/// run through `make test` (xcodebuild). The check mirrors the two places mlx looks (next to the executable and
/// the SwiftPM resource bundle) and errs on the side of skipping.
enum MLXTestEnvironment {
    static let skipEnvironmentKey = "MISAGO_SKIP_MLX_TESTS"

    static func skipIfMetallibUnavailable(file: StaticString = #filePath, line: UInt = #line) throws {
        if let flag = ProcessInfo.processInfo.environment[skipEnvironmentKey], flag.isEmpty != true, flag != "0" {
            throw XCTSkip("\(skipEnvironmentKey)=\(flag): skipping MLX tests", file: file, line: line)
        }
        if metallibURL() == nil {
            throw XCTSkip("MLX metallib not available under SwiftPM; run make test (xcodebuild)", file: file, line: line)
        }
    }

    static func metallibURL() -> URL? {
        let manager = FileManager.default
        var colocated: [URL] = []
        let testBundle = Bundle(for: MLXTestEnvironmentMarker.self)
        if let executableURL = testBundle.executableURL {
            colocated.append(executableURL.deletingLastPathComponent())
        }
        colocated.append(testBundle.bundleURL.deletingLastPathComponent())
        if let mainExecutable = Bundle.main.executableURL {
            colocated.append(mainExecutable.deletingLastPathComponent())
        }
        for directory in colocated {
            for relative in ["mlx.metallib", "Resources/mlx.metallib"] {
                let candidate = directory.appendingPathComponent(relative)
                if manager.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        var roots = colocated
        roots.append(testBundle.bundleURL)
        if let resourceURL = testBundle.resourceURL {
            roots.append(resourceURL)
        }
        for root in roots {
            let packageBundleURL = root.appendingPathComponent("mlx-swift_Cmlx.bundle")
            guard let packageBundle = Bundle(url: packageBundleURL), let resourceURL = packageBundle.resourceURL else {
                continue
            }
            let candidate = resourceURL.appendingPathComponent("default.metallib")
            if manager.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        return nil
    }
}

private final class MLXTestEnvironmentMarker {}
