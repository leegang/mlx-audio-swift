import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - Sinusoidal Position Embedding

private func sinusoidalPosEmb(_ x: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    precondition(dim % 2 == 0, "SinusoidalPosEmb requires dim to be even")
    let halfDim = dim / 2
    let emb = log(10000.0) / Float(halfDim - 1)
    let freqs = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * (-emb))
    var y = x
    if y.ndim < 1 {
        y = y.expandedDimensions(axis: 0)
    }
    let embVals = scale * y.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.sin(embVals), MLX.cos(embVals)], axis: -1)
}

// MARK: - Timestep Embedding

internal final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        let out = outDim ?? timeEmbedDim
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, out, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

// MARK: - VoxCPM2 DiT (V2 variant)

/// Diffusion Transformer for VoxCPM2.
/// Uses MiniCPM4Model as the Transformer backbone.
public final class VoxCPM2DiT: Module {
    let config: MiniCPM4Configuration
    let inChannels: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "decoder") var decoder: MiniCPM4Model

    @ModuleInfo(key: "time_mlp") var timeMLP: TimestepEmbedding
    @ModuleInfo(key: "delta_time_mlp") var deltaTimeMLP: TimestepEmbedding

    public init(config: MiniCPM4Configuration, inChannels: Int = 64) {
        self.config = config
        self.inChannels = inChannels

        self._inProj.wrappedValue = Linear(inChannels, config.hiddenSize, bias: true)
        self._condProj.wrappedValue = Linear(inChannels, config.hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(config.hiddenSize, inChannels, bias: true)

        self._timeMLP.wrappedValue = TimestepEmbedding(
            inChannels: config.hiddenSize,
            timeEmbedDim: config.hiddenSize
        )
        self._deltaTimeMLP.wrappedValue = TimestepEmbedding(
            inChannels: config.hiddenSize,
            timeEmbedDim: config.hiddenSize
        )

        self._decoder.wrappedValue = MiniCPM4Model(config)
    }

    /// Forward pass of DiT (V2 variant).
    ///
    /// - Parameters:
    ///   - x: [N, C, T] inputs
    ///   - mu: [N, hidden_size] hidden embedding
    ///   - t: [N] diffusion timesteps
    ///   - cond: [N, C, T_cond] prefix conditions
    ///   - dt: [N] delta timesteps
    /// - Returns: [N, C, T]
    public func callAsFunction(
        _ x: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        cond: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        let N = x.dim(0)
        let prefix = cond.dim(2)

        // Project inputs: [N, C, T] -> [N, T, hidden]
        var xProj = inProj(x.transposed(1, 2))

        // Project condition: [N, C, T_cond] -> [N, T_cond, hidden]
        let condProjVal = condProj(cond.transposed(1, 2))

        // Timestep embeddings
        var tEmb = sinusoidalPosEmb(t, dim: config.hiddenSize)
        tEmb = timeMLP(tEmb)
        var dtEmb = sinusoidalPosEmb(dt, dim: config.hiddenSize)
        dtEmb = deltaTimeMLP(dtEmb)
        tEmb = tEmb + dtEmb

        // mu: [N, hidden] -> [N, 1, hidden]
        let muView = mu.reshaped([N, 1, config.hiddenSize])

        // Concatenate: [mu, t, cond, x]
        let seq = MLX.concatenated([muView, tEmb.expandedDimensions(axis: 1), condProjVal, xProj], axis: 1)

        // Non-causal transformer
        var hidden = decoder(seq, mask: .none)  // [N, 1+1+T_cond+T, hidden]

        // Slice off prefix tokens (mu + t + cond)
        let startIdx = prefix + 2
        hidden = hidden[0..., startIdx..., 0...]  // [N, T, hidden]

        // Output projection and transpose back
        hidden = outProj(hidden)  // [N, T, C]
        return hidden.transposed(1, 2)  // [N, C, T]
    }
}
