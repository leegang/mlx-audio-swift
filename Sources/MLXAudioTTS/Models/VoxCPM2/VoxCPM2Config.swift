import Foundation
import MLXLMCommon

// MARK: - RoPE Scaling

public struct VoxCPM2RoPEScaling: Codable, Sendable {
    public var type: String
    public var longFactor: [Float]
    public var shortFactor: [Float]
    public var originalMaxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case type
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

// MARK: - LM Config (MiniCPM4)

public struct VoxCPM2LMConfig: Codable, Sendable {
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var maxPositionEmbeddings: Int
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var kvChannels: Int?
    public var ropeScaling: [String: StringOrNumber]?
    public var vocabSize: Int
    public var useMup: Bool
    public var scaleEmb: Float
    public var dimModelBase: Int
    public var scaleDepth: Float

    enum CodingKeys: String, CodingKey {
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case kvChannels = "kv_channels"
        case ropeScaling = "rope_scaling"
        case vocabSize = "vocab_size"
        case useMup = "use_mup"
        case scaleEmb = "scale_emb"
        case dimModelBase = "dim_model_base"
        case scaleDepth = "scale_depth"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        kvChannels = try c.decodeIfPresent(Int.self, forKey: .kvChannels)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 73448
        useMup = try c.decodeIfPresent(Bool.self, forKey: .useMup) ?? false
        scaleEmb = try c.decodeIfPresent(Float.self, forKey: .scaleEmb) ?? 1.0
        dimModelBase = try c.decodeIfPresent(Int.self, forKey: .dimModelBase) ?? 256
        scaleDepth = try c.decodeIfPresent(Float.self, forKey: .scaleDepth) ?? 1.0
    }
}

// MARK: - Encoder Config

public struct VoxCPM2EncoderConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var kvChannels: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 1024
        ffnDim = try c.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 4096
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
        kvChannels = try c.decodeIfPresent(Int.self, forKey: .kvChannels)
    }
}

// MARK: - CFM Config

public struct VoxCPM2CFMConfig: Codable, Sendable {
    public var sigmaMin: Float
    public var solver: String
    public var tScheduler: String
    public var inferenceCfgRate: Float

    enum CodingKeys: String, CodingKey {
        case sigmaMin = "sigma_min"
        case solver
        case tScheduler = "t_scheduler"
        case inferenceCfgRate = "inference_cfg_rate"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sigmaMin = try c.decodeIfPresent(Float.self, forKey: .sigmaMin) ?? 1e-6
        solver = try c.decodeIfPresent(String.self, forKey: .solver) ?? "euler"
        tScheduler = try c.decodeIfPresent(String.self, forKey: .tScheduler) ?? "log-norm"
        inferenceCfgRate = try c.decodeIfPresent(Float.self, forKey: .inferenceCfgRate) ?? 2.0
    }
}

// MARK: - DiT Config

public struct VoxCPM2DiTConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var kvChannels: Int?
    public var meanMode: Bool
    public var cfmConfig: VoxCPM2CFMConfig

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
        case meanMode = "mean_mode"
        case cfmConfig = "cfm_config"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 1024
        ffnDim = try c.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 4096
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 12
        kvChannels = try c.decodeIfPresent(Int.self, forKey: .kvChannels)
        meanMode = try c.decodeIfPresent(Bool.self, forKey: .meanMode) ?? false
        cfmConfig = try c.decode(VoxCPM2CFMConfig.self, forKey: .cfmConfig)
    }
}

// MARK: - Audio VAE Config

public struct VoxCPM2AudioVAEConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var srBinBoundaries: [Int]
    public var sampleRate: Int
    public var outSampleRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case srBinBoundaries = "sr_bin_boundaries"
        case sampleRate = "sample_rate"
        case outSampleRate = "out_sample_rate"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 128
        encoderRates = try c.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 5, 8, 8]
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        decoderDim = try c.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 2048
        decoderRates = try c.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [8, 6, 5, 2, 2, 2]
        srBinBoundaries = try c.decodeIfPresent([Int].self, forKey: .srBinBoundaries) ?? [20000, 30000, 40000]
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        outSampleRate = try c.decodeIfPresent(Int.self, forKey: .outSampleRate) ?? 48000
    }
}

// MARK: - Quantization Config

public struct VoxCPM2QuantizationConfig: Codable, Sendable {
    public var bits: Int
    public var groupSize: Int
    public var targets: [String]

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case targets
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
        groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        targets = try c.decodeIfPresent([String].self, forKey: .targets) ?? ["base_lm", "residual_lm"]
    }
}

// MARK: - Top-level Config

public struct VoxCPM2Config: Codable, Sendable {
    public var modelType: String
    public var architecture: String?
    public var lmConfig: VoxCPM2LMConfig
    public var patchSize: Int
    public var featDim: Int
    public var scalarQuantizationLatentDim: Int
    public var scalarQuantizationScale: Int
    public var residualLmNumLayers: Int
    public var residualLmNoRope: Bool
    public var encoderConfig: VoxCPM2EncoderConfig
    public var ditConfig: VoxCPM2DiTConfig
    public var audioVaeConfig: VoxCPM2AudioVAEConfig
    public var maxLength: Int
    public var quantization: VoxCPM2QuantizationConfig?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case lmConfig = "lm_config"
        case patchSize = "patch_size"
        case featDim = "feat_dim"
        case scalarQuantizationLatentDim = "scalar_quantization_latent_dim"
        case scalarQuantizationScale = "scalar_quantization_scale"
        case residualLmNumLayers = "residual_lm_num_layers"
        case residualLmNoRope = "residual_lm_no_rope"
        case encoderConfig = "encoder_config"
        case ditConfig = "dit_config"
        case audioVaeConfig = "audio_vae_config"
        case maxLength = "max_length"
        case quantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxcpm2"
        architecture = try c.decodeIfPresent(String.self, forKey: .architecture)
        lmConfig = try c.decode(VoxCPM2LMConfig.self, forKey: .lmConfig)
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 4
        featDim = try c.decodeIfPresent(Int.self, forKey: .featDim) ?? 64
        scalarQuantizationLatentDim = try c.decodeIfPresent(Int.self, forKey: .scalarQuantizationLatentDim) ?? 512
        scalarQuantizationScale = try c.decodeIfPresent(Int.self, forKey: .scalarQuantizationScale) ?? 9
        residualLmNumLayers = try c.decodeIfPresent(Int.self, forKey: .residualLmNumLayers) ?? 8
        residualLmNoRope = try c.decodeIfPresent(Bool.self, forKey: .residualLmNoRope) ?? true
        encoderConfig = try c.decode(VoxCPM2EncoderConfig.self, forKey: .encoderConfig)
        ditConfig = try c.decode(VoxCPM2DiTConfig.self, forKey: .ditConfig)
        audioVaeConfig = try c.decode(VoxCPM2AudioVAEConfig.self, forKey: .audioVaeConfig)
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? 8192

        let baseConfig = try? BaseConfiguration(from: decoder)
        quantization = try c.decodeIfPresent(VoxCPM2QuantizationConfig.self, forKey: .quantization)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public var sampleRate: Int {
        audioVaeConfig.outSampleRate
    }
}
