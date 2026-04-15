import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Scalar Quantization Layer

/// FSQ-like scalar quantization layer used in VoxCPM2.
///
/// Forward:
/// 1. Project input to latent_dim
/// 2. Apply tanh
/// 3. Round to discrete levels (inference) or straight-through estimator (training)
/// 4. Project back to out_dim
public final class ScalarQuantizationLayer: Module {
    let latentDim: Int
    let scale: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(inDim: Int, outDim: Int, latentDim: Int = 64, scale: Int = 9) {
        self.latentDim = latentDim
        self.scale = Float(scale)
        self._inProj.wrappedValue = Linear(inDim, latentDim, bias: true)
        self._outProj.wrappedValue = Linear(latentDim, outDim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inProj(x)
        h = MLX.tanh(h)
        h = MLX.round(h * scale) / scale
        return outProj(h)
    }
}
