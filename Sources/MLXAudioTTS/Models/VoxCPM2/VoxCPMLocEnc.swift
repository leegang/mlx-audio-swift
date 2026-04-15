import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - Local Encoder

/// VoxCPM Local Encoder: processes audio feature patches via a non-causal Transformer encoder.
///
/// Architecture:
/// - Input projection: [B, T, P, D] -> [B, T, P, hidden_size]
/// - Prepends a learnable special (CLS) token
/// - Runs a MiniCPM4Model encoder (non-causal)
/// - Returns CLS output for each time step: [B, T, hidden_size]
public final class VoxCPMLocEnc: Module {
    let config: MiniCPM4Configuration
    let inputDim: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "encoder") var encoder: MiniCPM4Model

    public init(config: MiniCPM4Configuration, inputDim: Int = 64) {
        self.config = config
        self.inputDim = inputDim

        self._inProj.wrappedValue = Linear(inputDim, config.hiddenSize, bias: true)
        self._specialToken.wrappedValue = MLXRandom.normal([1, 1, 1, config.hiddenSize])
        self._encoder.wrappedValue = MiniCPM4Model(config)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, P, D]
        let B = x.dim(0)
        let T = x.dim(1)

        var h = inProj(x)  // [B, T, P, hidden_size]
        let specialTokens = MLX.broadcast(specialToken, to: [B, T, 1, config.hiddenSize])
        h = MLX.concatenated([specialTokens, h], axis: 2)  // [B, T, P+1, hidden_size]
        h = h.reshaped([B * T, h.dim(2), config.hiddenSize])  // [(B*T), P+1, hidden_size]

        let outputs = encoder(h, mask: .none)  // [(B*T), P+1, hidden_size]
        let clsOutput = outputs[0..., 0, 0...]  // [(B*T), hidden_size]

        return clsOutput.reshaped([B, T, config.hiddenSize])  // [B, T, hidden_size]
    }
}
