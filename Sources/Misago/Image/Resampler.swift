import Foundation

/// Resamples a crop of an interleaved-channel image into a fixed output size: area averaging when shrinking
/// in both directions, bilinear otherwise. Samples outside the input are black. Used by the screen and strip
/// normalizers.
/// Read-only pixel samples in [0, 1]; lets the kernels run on bytes or floats without copying the image.
public protocol PixelSource {
    func value(_ index: Int) -> Float
    /// Copies `count` consecutive samples starting at `index` into `buffer` as floats.
    func fill(_ buffer: UnsafeMutablePointer<Float>, from index: Int, count: Int)
}

public struct FloatPixels: PixelSource {
    public let data: [Float]
    public init(_ data: [Float]) {
        self.data = data
    }
    @inline(__always)
    public func value(_ index: Int) -> Float {
        return data[index]
    }
    public func fill(_ buffer: UnsafeMutablePointer<Float>, from index: Int, count: Int) {
        data.withUnsafeBufferPointer { p in
            buffer.update(from: p.baseAddress! + index, count: count)
        }
    }
}

public struct BytePixels: PixelSource {
    public let data: [UInt8]
    public init(_ data: [UInt8]) {
        self.data = data
    }
    @inline(__always)
    public func value(_ index: Int) -> Float {
        return Float(data[index]) * (1.0 / 255.0)
    }
    public func fill(_ buffer: UnsafeMutablePointer<Float>, from index: Int, count: Int) {
        data.withUnsafeBufferPointer { p in
            let src = p.baseAddress! + index
            let scale = SIMD8<Float>(repeating: 1.0 / 255.0)
            var i = 0
            while i + 8 <= count {
                let bytes = UnsafeRawPointer(src + i).loadUnaligned(as: SIMD8<UInt8>.self)
                UnsafeMutableRawPointer(buffer + i).storeBytes(of: SIMD8<Float>(bytes) * scale, as: SIMD8<Float>.self)
                i += 8
            }
            while i < count {
                buffer[i] = Float(src[i]) * (1.0 / 255.0)
                i += 1
            }
        }
    }
}

public enum Resampler {
    /// `rgb` is `width * height * channels` samples in [0, 1], row-major. The crop (input pixels, may extend
    /// outside the image) is mapped onto `outWidth` x `outHeight` output pixels.
    public static func resample(
        rgb: [Float], width: Int, height: Int, channels: Int,
        cropX: Float, cropY: Float, cropWidth: Float, cropHeight: Float,
        outWidth: Int, outHeight: Int
    ) -> [Float] {
        return resample(source: FloatPixels(rgb), width: width, height: height, channels: channels, cropX: cropX, cropY: cropY, cropWidth: cropWidth, cropHeight: cropHeight, outWidth: outWidth, outHeight: outHeight)
    }

    /// Same on 8-bit samples (0...255), read directly without an intermediate float image.
    public static func resample(
        bytes: [UInt8], width: Int, height: Int, channels: Int,
        cropX: Float, cropY: Float, cropWidth: Float, cropHeight: Float,
        outWidth: Int, outHeight: Int
    ) -> [Float] {
        return resample(source: BytePixels(bytes), width: width, height: height, channels: channels, cropX: cropX, cropY: cropY, cropWidth: cropWidth, cropHeight: cropHeight, outWidth: outWidth, outHeight: outHeight)
    }

    public static func resample<S: PixelSource>(
        source rgb: S, width: Int, height: Int, channels: Int,
        cropX: Float, cropY: Float, cropWidth: Float, cropHeight: Float,
        outWidth: Int, outHeight: Int
    ) -> [Float] {
        var out = [Float](repeating: 0.0, count: outWidth * outHeight * channels)
        if outWidth <= 0 || outHeight <= 0 || cropWidth <= 0.0 || cropHeight <= 0.0 {
            return out
        }
        let scaleX = cropWidth / Float(outWidth)
        let scaleY = cropHeight / Float(outHeight)
        if 1.0 < scaleX && 1.0 < scaleY {
            areaAverage(rgb: rgb, width: width, height: height, channels: channels, originX: cropX, originY: cropY, scaleX: scaleX, scaleY: scaleY, outWidth: outWidth, outHeight: outHeight, into: &out)
        } else {
            bilinear(rgb: rgb, width: width, height: height, channels: channels, originX: cropX, originY: cropY, scaleX: scaleX, scaleY: scaleY, outWidth: outWidth, outHeight: outHeight, into: &out)
        }
        return out
    }

    /// Overlap weights of one output axis: for each output index, the input indices it covers and how much of
    /// each (partial pixels count by their overlap). Indices outside the input are dropped (they are black).
    private static func axisWeights(origin: Float, scale: Float, outCount: Int, inCount: Int) -> (starts: [Int], indices: [Int], weights: [Float]) {
        var starts = [Int](repeating: 0, count: outCount + 1)
        var indices: [Int] = []
        var weights: [Float] = []
        indices.reserveCapacity(outCount * (Int(scale) + 2))
        weights.reserveCapacity(outCount * (Int(scale) + 2))
        var o = 0
        while o < outCount {
            starts[o] = indices.count
            let a0 = origin + Float(o) * scale
            let a1 = a0 + scale
            var i = max(0, Int(a0.rounded(.down)))
            let end = min(inCount, Int(a1.rounded(.up)))
            while i < end {
                let w = min(a1, Float(i + 1)) - max(a0, Float(i))
                if 0.0 < w {
                    indices.append(i)
                    weights.append(w)
                }
                i += 1
            }
            o += 1
        }
        starts[outCount] = indices.count
        return (starts, indices, weights)
    }

    /// Shrinking: each output pixel is the mean of the input rectangle it covers, with partial pixels weighted
    /// by their overlap. Area outside the input is black and still counts toward the mean.
    ///
    /// Two passes, vertical first: every input row is converted to floats once and added (SIMD, contiguous)
    /// into the one or two output rows it overlaps; the horizontal pass then runs on just `outHeight` rows.
    private static func areaAverage<S: PixelSource>(
        rgb: S, width: Int, height: Int, channels ch: Int,
        originX: Float, originY: Float, scaleX: Float, scaleY: Float,
        outWidth outW: Int, outHeight outH: Int,
        into out: inout [Float]
    ) {
        let xs = axisWeights(origin: originX, scale: scaleX, outCount: outW, inCount: width)
        let ys = axisWeights(origin: originY, scale: scaleY, outCount: outH, inCount: height)
        // Invert the vertical weights: for each input row, the output rows it contributes to.
        var rowTargets = [[(row: Int, weight: Float)]](repeating: [], count: height)
        var oy = 0
        while oy < outH {
            var k = ys.starts[oy]
            while k < ys.starts[oy + 1] {
                rowTargets[ys.indices[k]].append((oy, ys.weights[k]))
                k += 1
            }
            oy += 1
        }
        let rowLen = width * ch
        // Pass 1: vertical sums [outH, width, ch].
        var vertical = [Float](repeating: 0.0, count: outH * rowLen)
        var rowBuffer = [Float](repeating: 0.0, count: rowLen)
        vertical.withUnsafeMutableBufferPointer { vp in
        rowBuffer.withUnsafeMutableBufferPointer { rp in
            let v = vp.baseAddress!
            let row = rp.baseAddress!
            var iy = 0
            while iy < height {
                let targets = rowTargets[iy]
                if targets.isEmpty != true {
                    rgb.fill(row, from: iy * rowLen, count: rowLen)
                    for target in targets {
                        let dst = v + target.row * rowLen
                        let w = SIMD8<Float>(repeating: target.weight)
                        var i = 0
                        while i + 8 <= rowLen {
                            let acc = UnsafeRawPointer(dst + i).loadUnaligned(as: SIMD8<Float>.self)
                            let src = UnsafeRawPointer(row + i).loadUnaligned(as: SIMD8<Float>.self)
                            UnsafeMutableRawPointer(dst + i).storeBytes(of: acc.addingProduct(src, w), as: SIMD8<Float>.self)
                            i += 8
                        }
                        while i < rowLen {
                            dst[i] += row[i] * target.weight
                            i += 1
                        }
                    }
                }
                iy += 1
            }
        }}
        // Pass 2: horizontal sums over the vertical rows, then scale by the cell area.
        let inv = 1.0 / (scaleX * scaleY)
        vertical.withUnsafeBufferPointer { vp in
        out.withUnsafeMutableBufferPointer { op in
        xs.starts.withUnsafeBufferPointer { sp in
        xs.indices.withUnsafeBufferPointer { ip in
        xs.weights.withUnsafeBufferPointer { wp in
            let v = vp.baseAddress!
            let o = op.baseAddress!
            let starts = sp.baseAddress!
            let indices = ip.baseAddress!
            let weights = wp.baseAddress!
            var oy = 0
            while oy < outH {
                let src = v + oy * rowLen
                let dst = o + oy * outW * ch
                var ox = 0
                while ox < outW {
                    let kEnd = starts[ox + 1]
                    var c = 0
                    while c < ch {
                        var acc: Float = 0.0
                        var k = starts[ox]
                        while k < kEnd {
                            acc += src[indices[k] * ch + c] * weights[k]
                            k += 1
                        }
                        dst[ox * ch + c] = clamp01(acc * inv)
                        c += 1
                    }
                    ox += 1
                }
                oy += 1
            }
        }}}}}
    }

    /// Enlarging (or mixed): bilinear interpolation at each output pixel center. An output pixel whose center
    /// lies outside the input is black; inside, samples are clamped to the input edge.
    private static func bilinear<S: PixelSource>(
        rgb: S, width: Int, height: Int, channels ch: Int,
        originX: Float, originY: Float, scaleX: Float, scaleY: Float,
        outWidth outW: Int, outHeight outH: Int,
        into out: inout [Float]
    ) {
        var oy = 0
        while oy < outH {
            let cy = originY + (Float(oy) + 0.5) * scaleY
            if cy < 0.0 || Float(height) <= cy {
                oy += 1
                continue
            }
            let sy = min(max(cy - 0.5, 0.0), Float(height - 1))
            let fy = sy.rounded(.down)
            let ty = sy - fy
            let iy0 = Int(fy)
            let iy1 = min(height - 1, iy0 + 1)
            var ox = 0
            while ox < outW {
                let cx = originX + (Float(ox) + 0.5) * scaleX
                if cx < 0.0 || Float(width) <= cx {
                    ox += 1
                    continue
                }
                let sx = min(max(cx - 0.5, 0.0), Float(width - 1))
                let fx = sx.rounded(.down)
                let tx = sx - fx
                let ix0 = Int(fx)
                let ix1 = min(width - 1, ix0 + 1)
                var c = 0
                while c < ch {
                    let top = rgb.value((iy0 * width + ix0) * ch + c) * (1.0 - tx) + rgb.value((iy0 * width + ix1) * ch + c) * tx
                    let bottom = rgb.value((iy1 * width + ix0) * ch + c) * (1.0 - tx) + rgb.value((iy1 * width + ix1) * ch + c) * tx
                    out[(oy * outW + ox) * ch + c] = clamp01(top * (1.0 - ty) + bottom * ty)
                    c += 1
                }
                ox += 1
            }
            oy += 1
        }
    }

    @inline(__always)
    static func clamp01(_ v: Float) -> Float {
        if v < 0.0 {
            return 0.0
        }
        if 1.0 < v {
            return 1.0
        }
        return v
    }

    /// Converts 8-bit samples to [0, 1].
    public static func floats(_ bytes: [UInt8]) -> [Float] {
        var values = [Float](repeating: 0.0, count: bytes.count)
        var i = 0
        while i < bytes.count {
            values[i] = Float(bytes[i]) * (1.0 / 255.0)
            i += 1
        }
        return values
    }
}
