import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast
import MLXLMCommon

// MARK: - Configuration

public struct MiniCPM4Configuration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var ropeTheta: Float
    public var headDim: Int
    public var ropeScaling: [String: StringOrNumber]?
    public var tieWordEmbeddings: Bool
    public var maxPositionEmbeddings: Int
    public var useMup: Bool
    public var scaleDepth: Float
    public var noRope: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useMup = "use_mup"
        case scaleDepth = "scale_depth"
        case noRope = "no_rope"
    }

    public init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        vocabularySize: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        maxPositionEmbeddings: Int = 32768,
        useMup: Bool = false,
        scaleDepth: Float = 1.0,
        noRope: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.vocabularySize = vocabularySize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.useMup = useMup
        self.scaleDepth = scaleDepth
        self.noRope = noRope
    }
}

// MARK: - Attention

public class MiniCPMAttention: Module {
    let args: MiniCPM4Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: OffsetLayer?

    public init(_ args: MiniCPM4Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim

        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        if args.noRope {
            self.rope = nil
        } else {
            self.rope = initializeRope(
                dims: headDim,
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings
            )
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            queries = rope(queries, offset: cache?.offset ?? 0)
            keys = rope(keys, offset: cache?.offset ?? 0)
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

public class MiniCPMMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder Layer

public class MiniCPMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MiniCPMAttention
    let mlp: MiniCPMMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let scaleDepth: Float
    let useMup: Bool
    let numHiddenLayers: Int

    public init(_ args: MiniCPM4Configuration) {
        self._attention.wrappedValue = MiniCPMAttention(args)
        self.mlp = MiniCPMMLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.scaleDepth = args.scaleDepth
        self.useMup = args.useMup
        self.numHiddenLayers = args.hiddenLayers
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var hiddenStates = x

        var residual = hiddenStates
        hiddenStates = inputLayerNorm(hiddenStates)
        hiddenStates = attention(hiddenStates, mask: mask, cache: cache)
        hiddenStates = useMup
            ? residual + hiddenStates * (scaleDepth / sqrt(Float(numHiddenLayers)))
            : residual + hiddenStates

        residual = hiddenStates
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = mlp(hiddenStates)
        hiddenStates = useMup
            ? residual + hiddenStates * (scaleDepth / sqrt(Float(numHiddenLayers)))
            : residual + hiddenStates

        return hiddenStates
    }
}

// MARK: - Model

public class MiniCPM4Model: Module {
    let config: MiniCPM4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [MiniCPMDecoderLayer]
    let norm: RMSNorm

    public init(_ args: MiniCPM4Configuration) {
        self.config = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: max(1, args.vocabularySize),
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers)
            .map { _ in MiniCPMDecoderLayer(args) }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        var h = config.vocabularySize > 0 ? embedTokens(inputs) : inputs
        let resolvedMask: MLXFast.ScaledDotProductAttentionMaskMode = mask ?? createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: resolvedMask, cache: cache?[i])
        }

        return norm(h)
    }

    public func forwardWithEmbeddings(
        inputsEmbeds: MLXArray,
        cache: [KVCache]? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> MLXArray {
        var h = inputsEmbeds
        let resolvedMask: MLXFast.ScaledDotProductAttentionMaskMode = mask ?? createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: resolvedMask, cache: cache?[i])
        }

        return norm(h)
    }

    public func getEmbeddings(for inputIds: MLXArray) -> MLXArray {
        config.vocabularySize > 0 ? embedTokens(inputIds) : inputIds
    }
}
