import Foundation
import Vapor
import Misago
import ImageDecoding

/// HTTP front end for the screen reader: `POST /api/read` with an image (multipart field `image`, or the raw
/// image as the body) returns the class, its calibrated probability, the number read and timings. Static files
/// (the upload page) are served from `MISAGO_PUBLIC`.
///
/// Environment: MISAGO_WEIGHTS (classifier JSON), MISAGO_DIGIT_WEIGHTS (reader JSON), MISAGO_REGIONS
/// (regions JSON), MISAGO_PUBLIC (static directory, default .tmp/public),
/// PORT (default 8080, set by Cloud Run).
struct ReadResponse: Content {
    let `class`: String
    let probability: Float
    /// Screen probabilities (a softmax among the screens of all titles).
    let probabilities: [String: Float]
    /// Title of the winning screen and the independent probability that the image is a screen of it.
    let title: String
    let title_probability: Float
    /// Independent probability of every title: all near 0 for an image of nothing the model knows.
    let titles: [String: Float]
    let number: Int
    /// 2 = win, 1 = loss, 0 = draw, −1 = not a result screen (class names ending in _win / _lose / _draw).
    let outcome: Int
    /// 1 during a match (class names ending in _battle or containing _battle_), 0 otherwise.
    let battle: Int
    let digits: String?
    let confidence: Float?
    /// Clockwise rotation (0, 90, 180, 270 degrees) that turned the image upright.
    let rotation: Int
    let ms: Double
    /// Server-side stage times: decode, normalize (rotation to landscape and whole-screen downscale), classify
    /// (class and orientation), extract (upright crop), read (digits).
    let timings_ms: [String: Double]
    let width: Int
    let height: Int
}

struct ErrorResponse: Content {
    let error: String
}

struct Upload: Content {
    var image: File
}

@main
struct MisagoAPI {
    static func main() async throws {
        var env = try Environment.detect()
        try LoggingSystem.bootstrap(from: &env)
        let app = try await Application.make(env)
        do {
            try configure(app)
            try await app.execute()
        } catch {
            app.logger.report(error: error)
            try? await app.asyncShutdown()
            throw error
        }
        try await app.asyncShutdown()
    }

    static func environment(_ key: String, default value: String) -> String {
        return Environment.get(key) ?? value
    }

    static func configure(_ app: Application) throws {
        let weightsPath = environment("MISAGO_WEIGHTS", default: ".tmp/models/classifier.json")
        let digitWeightsPath = environment("MISAGO_DIGIT_WEIGHTS", default: ".tmp/models/digits.json")
        let regionsPath = environment("MISAGO_REGIONS", default: ".tmp/models/regions.json")
        let publicDirectory = environment("MISAGO_PUBLIC", default: ".tmp/public")
        app.http.server.configuration.hostname = environment("HOST", default: "0.0.0.0")
        app.http.server.configuration.port = Int(environment("PORT", default: "8080")) ?? 8080
        app.routes.defaultMaxBodySize = "32mb"

        let classifier = try ImageClassifier(weights: try HawkWeights.load(from: URL(fileURLWithPath: weightsPath)))
        let digits = try DigitReader(weights: try DigitReaderWeights.load(from: URL(fileURLWithPath: digitWeightsPath)))
        let regions = try RegionTable.load(from: URL(fileURLWithPath: regionsPath))
        let reader = ScreenReader(classifier: classifier, digitReader: digits, regions: regions)
        app.logger.info("models loaded: classes \(classifier.classes.names.joined(separator: ", ")), regions \(regions.regions.keys.sorted().joined(separator: ", "))")

        if FileManager.default.fileExists(atPath: publicDirectory) {
            app.middleware.use(FileMiddleware(publicDirectory: publicDirectory, defaultFile: "index.html"))
        } else {
            app.logger.warning("public directory \(publicDirectory) not found; only the API is served")
        }

        app.get("healthz") { _ in
            return "ok"
        }

        app.on(.POST, "api", "read", body: .collect(maxSize: "32mb")) { req async throws -> Response in
            let bytes: Data
            if req.headers.contentType?.type == "multipart" {
                let upload = try req.content.decode(Upload.self)
                bytes = Data(buffer: upload.image.data)
            } else if let body = req.body.data {
                bytes = Data(buffer: body)
            } else {
                throw Abort(.badRequest, reason: "send the image as multipart field `image` or as the request body")
            }
            func now() -> Double {
                return Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000.0
            }
            func round2(_ v: Double) -> Double {
                return (v * 100).rounded() / 100
            }
            let decodeStart = now()
            let decoded: (width: Int, height: Int, rgb: [UInt8])
            do {
                decoded = try ImageDecoder.decodeRGB(bytes)
            } catch {
                throw Abort(.unprocessableEntity, reason: "cannot decode image: \(error)")
            }
            let decodeMs = now() - decodeStart
            // The stages are run separately here (same work as `ScreenReader.read`) so each can be timed.
            let start = now()
            let input = try reader.landscape(width: decoded.width, height: decoded.height, rgb: decoded.rgb)
            let normalizeMs = now() - start
            let classifyStart = now()
            let oriented = reader.classify(input)
            let classifyMs = now() - classifyStart
            let extractStart = now()
            let strip = try reader.strip(input, verdict: oriented.verdict, rotation: oriented.rotation)
            let extractMs = now() - extractStart
            var readMs = 0.0
            var reading: DigitReading? = nil
            if let strip = strip {
                let readStart = now()
                reading = digits.read(strip)
                readMs = now() - readStart
            }
            let result = ScreenReading(verdict: oriented.verdict, reading: reading, rotation: oriented.rotation)
            let ms = now() - start
            req.logger.info("read \(result.verdict.className) number=\(result.number) rotation=\(result.rotation) battle=\(result.battle) outcome=\(result.outcome) decode=\(round2(decodeMs))ms total=\(round2(ms))ms (normalize \(round2(normalizeMs)), classify \(round2(classifyMs)), extract \(round2(extractMs)), read \(round2(readMs)))")
            var probabilities: [String: Float] = [:]
            var c = 0
            while c < result.verdict.probabilities.count {
                probabilities[classifier.classes[c]] = result.verdict.probabilities[c]
                c += 1
            }
            let payload = ReadResponse(
                class: result.verdict.className,
                probability: result.verdict.probability,
                probabilities: probabilities,
                title: result.verdict.title,
                title_probability: result.verdict.titleProbability,
                titles: result.verdict.titleProbabilities,
                number: result.number,
                outcome: result.outcome,
                battle: result.battle,
                digits: result.reading?.digits,
                confidence: result.reading?.confidence,
                rotation: result.rotation,
                ms: round2(ms),
                timings_ms: ["decode": round2(decodeMs), "normalize": round2(normalizeMs), "classify": round2(classifyMs), "extract": round2(extractMs), "read": round2(readMs)],
                width: decoded.width,
                height: decoded.height
            )
            let response = Response(status: .ok)
            try response.content.encode(payload, as: .json)
            return response
        }
    }
}
