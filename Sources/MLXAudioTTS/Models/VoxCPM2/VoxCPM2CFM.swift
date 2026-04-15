import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - VoxCPM2 CFM

/// Conditional Flow Matching solver for VoxCPM2 inference.
///
/// Implements the Euler solver from the Python `UnifiedCFM` with
/// classifier-free guidance (CFG) support.
public final class VoxCPM2CFM: Module {
    let inChannels: Int
    let meanMode: Bool
    let inferenceCfgRate: Float

    @ModuleInfo(key: "estimator") var estimator: VoxCPM2DiT

    public init(
        inChannels: Int,
        inferenceCfgRate: Float,
        estimator: VoxCPM2DiT,
        meanMode: Bool = false
    ) {
        self.inChannels = inChannels
        self.inferenceCfgRate = inferenceCfgRate
        self.meanMode = meanMode
        self._estimator.wrappedValue = estimator
    }

    /// Generate latent features via Euler solver.
    ///
    /// - Parameters:
    ///   - mu: [B, hidden_size] conditional embedding
    ///   - nTimesteps: number of Euler steps (default 10)
    ///   - patchSize: number of patches per time step
    ///   - cond: [B, inChannels, condLen] prefix condition
    ///   - cfgValue: classifier-free guidance scale (default 2.0)
    ///   - temperature: noise temperature (default 1.0)
    ///   - swaySamplingCoef: sway sampling coefficient (default 1.0)
    /// - Returns: [B, inChannels, patchSize] generated features
    public func generate(
        mu: MLXArray,
        nTimesteps: Int = 10,
        patchSize: Int,
        cond: MLXArray,
        cfgValue: Float = 2.0,
        temperature: Float = 1.0,
        swaySamplingCoef: Float = 1.0
    ) -> MLXArray {
        let b = mu.dim(0)

        // Initialize noise
        var x = MLXRandom.normal([b, inChannels, patchSize]) * temperature

        // Build t_span: linspace from 1 to 0
        var tSpan = MLXArray(stride(from: 0, through: nTimesteps, by: 1)).asType(.float32) / Float(nTimesteps)
        tSpan = 1.0 - tSpan  // [1.0, ..., 0.0]

        // Apply sway sampling
        let sway = swaySamplingCoef * (MLX.cos(Float.pi / 2.0 * tSpan) - 1.0 + tSpan)
        tSpan = tSpan + sway

        var t = tSpan[0].item(Float.self)
        var dt = t - tSpan[1].item(Float.self)

        for step in 1 ..< nTimesteps + 1 {
            let dphiDt: MLXArray

            if cfgValue > 0 {
                // Classifier-free guidance: duplicate inputs
                let xIn = MLX.concatenated([x, x], axis: 0)              // [2B, C, T]
                let muIn = MLX.concatenated([mu, MLXArray.zeros(mu.shape)], axis: 0)  // [2B, hidden]
                let tIn = MLXArray([Float](repeating: t, count: 2 * b))  // [2B]
                let dtIn = MLXArray([Float](repeating: meanMode ? dt : 0.0, count: 2 * b))  // [2B]
                let condIn = MLX.concatenated([cond, cond], axis: 0)     // [2B, C, condLen]

                var out = estimator(xIn, mu: muIn, t: tIn, cond: condIn, dt: dtIn)  // [2B, C, T]

                let condOut = out[0 ..< b]           // [B, C, T]
                let uncondOut = out[b ..< (2 * b)]   // [B, C, T]

                // Optimized scale (zero-star)
                let positiveFlat = condOut.reshaped([b, -1])
                let negativeFlat = uncondOut.reshaped([b, -1])
                let dotProduct = (positiveFlat * negativeFlat).sum(axis: 1, keepDims: true)
                let squaredNorm = (negativeFlat * negativeFlat).sum(axis: 1, keepDims: true) + 1e-8
                let stStar = dotProduct / squaredNorm
                let scale = stStar.reshaped([b, 1, 1])

                dphiDt = uncondOut * scale + cfgValue * (condOut - uncondOut * scale)
            } else {
                let tIn = MLXArray([Float](repeating: t, count: b))
                let dtIn = MLXArray([Float](repeating: meanMode ? dt : 0.0, count: b))
                dphiDt = estimator(x, mu: mu, t: tIn, cond: cond, dt: dtIn)
            }

            x = x - dt * dphiDt

            if step < nTimesteps {
                t = tSpan[step].item(Float.self)
                let nextT = tSpan[step + 1].item(Float.self)
                dt = t - nextT
            }
        }

        return x
    }
}
