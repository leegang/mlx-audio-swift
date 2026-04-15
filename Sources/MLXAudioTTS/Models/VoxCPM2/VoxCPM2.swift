import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - VoxCPM2 Model

/// VoxCPM2: A 2B-parameter tokenizer-free diffusion autoregressive TTS model.
///
/// Architecture:
/// - Base LM: 28-layer MiniCPM4 with LongRoPE
/// - Residual LM: 8-layer MiniCPM4 (no RoPE)
/// - Local Encoder: 12-layer Transformer for text/condition encoding
/// - DiT + CFM: Diffusion Transformer with Conditional Flow Matching
/// - Audio VAE V2: 16kHz -> 48kHz asymmetric encode/decode
public final class VoxCPM2Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: VoxCPM2Config

    @ModuleInfo(key: "base_lm") private var baseLM: MiniCPM4Model
    @ModuleInfo(key: "residual_lm") private var residualLM: MiniCPM4Model
    @ModuleInfo(key: "feat_encoder") private var featEncoder: VoxCPMLocEnc
    @ModuleInfo(key: "feat_decoder") private var featDecoder: VoxCPM2CFM
    @ModuleInfo(key: "audio_vae") private var audioVAE: VoxCPM2AudioVAE

    // Projection layers
    @ModuleInfo(key: "enc_to_lm_proj") private var encToLMProj: Linear
    @ModuleInfo(key: "lm_to_dit_proj") private var lmToDitProj: Linear
    @ModuleInfo(key: "res_to_dit_proj") private var resToDitProj: Linear
    @ModuleInfo(key: "fusion_concat_proj") private var fusionConcatProj: Linear

    // FSQ layer
    @ModuleInfo(key: "fsq_layer") private var fsqLayer: ScalarQuantizationLayer

    // Stop predictor
    @ModuleInfo(key: "stop_proj") private var stopProj: Linear
    @ModuleInfo(key: "stop_head") private var stopHead: Linear

    public var tokenizer: Tokenizer?

    // Special token IDs (matching Python implementation)
    private let audioStartToken: Int32 = 101
    private let audioEndToken: Int32 = 102
    private let refAudioStartToken: Int32 = 103
    private let refAudioEndToken: Int32 = 104

    public var sampleRate: Int { config.sampleRate }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 4096,
            temperature: 1.0,
            topP: 0.95,
            repetitionPenalty: 1.0
        )
    }

    public init(config: VoxCPM2Config) throws {
        self.config = config

        // Base LM: 28-layer MiniCPM4
        let baseConfig = MiniCPM4Configuration(
            hiddenSize: config.lmConfig.hiddenSize,
            hiddenLayers: config.lmConfig.numHiddenLayers,
            intermediateSize: config.lmConfig.intermediateSize,
            attentionHeads: config.lmConfig.numAttentionHeads,
            kvHeads: config.lmConfig.numKeyValueHeads,
            headDim: config.lmConfig.kvChannels ?? (config.lmConfig.hiddenSize / config.lmConfig.numAttentionHeads),
            vocabularySize: config.lmConfig.vocabSize,
            rmsNormEps: config.lmConfig.rmsNormEps,
            ropeTheta: config.lmConfig.ropeTheta,
            ropeScaling: config.lmConfig.ropeScaling,
            tieWordEmbeddings: true,
            maxPositionEmbeddings: config.lmConfig.maxPositionEmbeddings,
            useMup: config.lmConfig.useMup,
            scaleDepth: config.lmConfig.scaleDepth,
            noRope: false
        )
        self._baseLM.wrappedValue = MiniCPM4Model(baseConfig)

        // Residual LM: 8-layer, no vocab, no RoPE
        let residualConfig = MiniCPM4Configuration(
            hiddenSize: config.lmConfig.hiddenSize,
            hiddenLayers: config.residualLmNumLayers,
            intermediateSize: config.lmConfig.intermediateSize,
            attentionHeads: config.lmConfig.numAttentionHeads,
            kvHeads: config.lmConfig.numKeyValueHeads,
            headDim: config.lmConfig.kvChannels ?? (config.lmConfig.hiddenSize / config.lmConfig.numAttentionHeads),
            vocabularySize: 0,
            rmsNormEps: config.lmConfig.rmsNormEps,
            ropeTheta: config.lmConfig.ropeTheta,
            ropeScaling: config.lmConfig.ropeScaling,
            tieWordEmbeddings: true,
            maxPositionEmbeddings: config.lmConfig.maxPositionEmbeddings,
            useMup: config.lmConfig.useMup,
            scaleDepth: config.lmConfig.scaleDepth,
            noRope: config.residualLmNoRope
        )
        self._residualLM.wrappedValue = MiniCPM4Model(residualConfig)

        // Local Encoder (12-layer Transformer, hidden=1024)
        let encoderLMConfig = MiniCPM4Configuration(
            hiddenSize: config.encoderConfig.hiddenDim,
            hiddenLayers: config.encoderConfig.numLayers,
            intermediateSize: config.encoderConfig.ffnDim,
            attentionHeads: config.encoderConfig.numHeads,
            kvHeads: config.lmConfig.numKeyValueHeads,
            headDim: config.encoderConfig.kvChannels ?? (config.encoderConfig.hiddenDim / config.encoderConfig.numHeads),
            vocabularySize: 0,
            rmsNormEps: config.lmConfig.rmsNormEps,
            ropeTheta: config.lmConfig.ropeTheta,
            tieWordEmbeddings: true,
            maxPositionEmbeddings: config.lmConfig.maxPositionEmbeddings,
            useMup: false,
            scaleDepth: 1.0,
            noRope: false
        )
        self._featEncoder.wrappedValue = VoxCPMLocEnc(
            config: encoderLMConfig,
            inputDim: config.featDim
        )

        // DiT + CFM
        let ditLMConfig = MiniCPM4Configuration(
            hiddenSize: config.ditConfig.hiddenDim,
            hiddenLayers: config.ditConfig.numLayers,
            intermediateSize: config.ditConfig.ffnDim,
            attentionHeads: config.ditConfig.numHeads,
            kvHeads: config.lmConfig.numKeyValueHeads,
            headDim: config.ditConfig.kvChannels ?? (config.ditConfig.hiddenDim / config.ditConfig.numHeads),
            vocabularySize: 0,
            rmsNormEps: config.lmConfig.rmsNormEps,
            ropeTheta: config.lmConfig.ropeTheta,
            tieWordEmbeddings: true,
            maxPositionEmbeddings: config.lmConfig.maxPositionEmbeddings,
            useMup: false,
            scaleDepth: 1.0,
            noRope: false
        )
        let ditEstimator = VoxCPM2DiT(config: ditLMConfig, inChannels: config.featDim)
        self._featDecoder.wrappedValue = VoxCPM2CFM(
            inChannels: config.featDim,
            inferenceCfgRate: config.ditConfig.cfmConfig.inferenceCfgRate,
            estimator: ditEstimator,
            meanMode: config.ditConfig.meanMode
        )

        // Audio VAE V2
        self._audioVAE.wrappedValue = VoxCPM2AudioVAE(config: config.audioVaeConfig)

        // Projection layers
        self._encToLMProj.wrappedValue = Linear(
            inputDimensions: config.encoderConfig.hiddenDim,
            outputDimensions: config.lmConfig.hiddenSize,
            bias: true
        )
        self._lmToDitProj.wrappedValue = Linear(
            inputDimensions: config.lmConfig.hiddenSize,
            outputDimensions: config.ditConfig.hiddenDim,
            bias: true
        )
        self._resToDitProj.wrappedValue = Linear(
            inputDimensions: config.lmConfig.hiddenSize,
            outputDimensions: config.ditConfig.hiddenDim,
            bias: true
        )
        self._fusionConcatProj.wrappedValue = Linear(
            inputDimensions: config.lmConfig.hiddenSize * 2,
            outputDimensions: config.lmConfig.hiddenSize,
            bias: true
        )

        // FSQ layer
        self._fsqLayer.wrappedValue = ScalarQuantizationLayer(
            inDim: config.lmConfig.hiddenSize,
            outDim: config.lmConfig.hiddenSize,
            latentDim: config.scalarQuantizationLatentDim,
            scale: config.scalarQuantizationScale
        )

        // Stop predictor
        self._stopProj.wrappedValue = Linear(
            inputDimensions: config.lmConfig.hiddenSize,
            outputDimensions: config.lmConfig.hiddenSize,
            bias: true
        )
        self._stopHead.wrappedValue = Linear(
            inputDimensions: config.lmConfig.hiddenSize,
            outputDimensions: 2,
            bias: false
        )
    }

    // MARK: - Audio Encoding

    private func encodeWav(_ audio: MLXArray) -> MLXArray {
        // audio: [samples] at 16kHz
        var wav = audio
        if wav.ndim == 1 {
            wav = wav.expandedDimensions(axis: 0) // [1, samples]
        }
        let latent = audioVAE.encode(wav) // [1, latentDim, T']
        let B = latent.dim(0)
        let D = latent.dim(1)
        let T = latent.dim(2)
        let P = config.patchSize
        return latent.reshaped([B, D, T / P, P]).transposed(0, 2, 3, 1) // [B, T/P, P, D]
    }

    private func makeRefPrefix(refFeat: MLXArray) -> (tokens: MLXArray, feats: MLXArray, textMask: MLXArray, audioMask: MLXArray) {
        let refLen = refFeat.dim(0)
        let z1 = MLX.zeros([1, config.patchSize, config.audioVaeConfig.latentDim])
        let tokens = MLXArray([refAudioStartToken] + [Int32](repeating: 0, count: refLen) + [refAudioEndToken])
        let feats = MLX.concatenated([z1, refFeat, z1], axis: 0)
        let tMask = MLXArray([1] + [Int32](repeating: 0, count: refLen) + [1]).asType(.int32)
        var aMaskArr = [Int32](repeating: 1, count: refLen + 2)
        aMaskArr[0] = 0
        aMaskArr[refLen + 1] = 0
        let aMask = MLXArray(aMaskArr).asType(.int32)
        return (tokens, feats, tMask, aMask)
    }

    // MARK: - Generation

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        let inputs = try prepareGenerationInputs(text: text, refAudio: refAudio)

        let result = try inference(
            textToken: inputs.textToken,
            textMask: inputs.textMask,
            audioFeat: inputs.audioFeat,
            audioMask: inputs.audioMask,
            maxLen: min(256, generationParameters.maxTokens ?? 256),
            inferenceTimesteps: 5,
            cfgValue: config.ditConfig.cfmConfig.inferenceCfgRate
        )

        let decodeAudio = audioVAE.decode(result.featPred, srCond: config.audioVaeConfig.outSampleRate)
        return decodeAudio.squeezed(axis: 0).squeezed(axis: 0) // [samples]
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.generationFailed("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                let inputs = try self.prepareGenerationInputs(text: text, refAudio: refAudio)

                let prefillTime = Date().timeIntervalSince(startTime)
                let generateStartTime = Date()
                var tokenCount = 0

                let streamPrefixLen = 4
                let samplesPerPatch = self.config.patchSize * self.audioVAE.decodeChunkSize
                var rollingLatents: [MLXArray] = []
                var previousDecodedSamples: Int = 0
                var hasYielded = false

                try self.inferenceStream(
                    textToken: inputs.textToken,
                    textMask: inputs.textMask,
                    audioFeat: inputs.audioFeat,
                    audioMask: inputs.audioMask,
                    maxLen: min(256, generationParameters.maxTokens ?? 256),
                    inferenceTimesteps: 10,
                    cfgValue: self.config.ditConfig.cfmConfig.inferenceCfgRate,
                    streamingPrefixLen: streamPrefixLen
                ) { patchFeat in
                    tokenCount += 1

                    // patchFeat: [1, P, D]
                    rollingLatents.append(patchFeat)
                    if rollingLatents.count > streamPrefixLen {
                        rollingLatents.removeFirst()
                    }

                    // Concatenate rolling latents: [B, N, P, D] -> [B, D, N*P]
                    let latentSeq = MLX.concatenated(rollingLatents, axis: 1) // [B, N, P, D]
                    let B = latentSeq.dim(0)
                    let N = latentSeq.dim(1)
                    let P = latentSeq.dim(2)
                    let D = latentSeq.dim(3)
                    let latent = latentSeq.reshaped([B, N * P, D]).transposed(0, 2, 1) // [B, D, N*P]

                    let decoded = self.audioVAE.decode(latent, srCond: self.config.audioVaeConfig.outSampleRate)
                    // decoded: [1, 1, samples]
                    let totalSamples = decoded.dim(-1)
                    let newSamples = max(0, totalSamples - previousDecodedSamples)

                    if newSamples > 0 {
                        let chunk = decoded[0..., 0..., (totalSamples - newSamples)..<totalSamples].squeezed(axis: 0).squeezed(axis: 0)
                        continuation.yield(.audio(chunk))
                        hasYielded = true
                    }
                    previousDecodedSamples = totalSamples
                }

                let generateTime = Date().timeIntervalSince(generateStartTime)
                let info = AudioGenerationInfo(
                    promptTokenCount: inputs.textToken.dim(1),
                    generationTokenCount: tokenCount,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: generateTime > 0 ? Double(tokenCount) / generateTime : 0,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9
                )
                continuation.yield(.info(info))

                if !hasYielded {
                    continuation.yield(.audio(MLXArray.zeros([0])))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    private struct GenerationInputs {
        let textToken: MLXArray
        let textMask: MLXArray
        let audioFeat: MLXArray
        let audioMask: MLXArray
    }

    private func prepareGenerationInputs(text: String, refAudio: MLXArray?) throws -> GenerationInputs {
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let targetTextTokens = try tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) }
        let targetTextToken = MLXArray(targetTextTokens + [audioStartToken]).asType(.int32)
        let targetTextLength = targetTextToken.shape[0]

        var refFeat: MLXArray?
        if let refAudio = refAudio {
            let encoded = encodeWav(refAudio)
            refFeat = encoded[0]
        }

        var textToken: MLXArray
        var audioFeat: MLXArray
        var textMask: MLXArray
        var audioMask: MLXArray

        if let refFeat = refFeat {
            let (refTokens, refFeats, refTMask, refAMask) = makeRefPrefix(refFeat: refFeat)
            let textPadFeat = MLX.zeros([targetTextLength, config.patchSize, config.audioVaeConfig.latentDim])
            textToken = MLX.concatenated([refTokens, targetTextToken])
            audioFeat = MLX.concatenated([refFeats, textPadFeat], axis: 0)
            textMask = MLX.concatenated([refTMask, MLXArray.ones([targetTextLength], type: Int32.self)])
            audioMask = MLX.concatenated([refAMask, MLXArray.zeros([targetTextLength], type: Int32.self)])
        } else {
            let textPadFeat = MLX.zeros([targetTextLength, config.patchSize, config.audioVaeConfig.latentDim])
            textToken = targetTextToken
            audioFeat = textPadFeat
            textMask = MLXArray.ones([targetTextLength], type: Int32.self)
            audioMask = MLXArray.zeros([targetTextLength], type: Int32.self)
        }

        return GenerationInputs(
            textToken: textToken.expandedDimensions(axis: 0),
            textMask: textMask.expandedDimensions(axis: 0).asType(.float32),
            audioFeat: audioFeat.expandedDimensions(axis: 0),
            audioMask: audioMask.expandedDimensions(axis: 0).asType(.float32)
        )
    }

    private struct InferenceResult {
        let featPred: MLXArray
        let generatedFeat: MLXArray
        let contextLen: Int
    }

    private func inference(
        textToken: MLXArray,
        textMask: MLXArray,
        audioFeat: MLXArray,
        audioMask: MLXArray,
        maxLen: Int,
        inferenceTimesteps: Int,
        cfgValue: Float
    ) throws -> InferenceResult {
        var predFeatSeq: [MLXArray] = []
        try inferenceStream(
            textToken: textToken,
            textMask: textMask,
            audioFeat: audioFeat,
            audioMask: audioMask,
            maxLen: maxLen,
            inferenceTimesteps: inferenceTimesteps,
            cfgValue: cfgValue,
            streamingPrefixLen: 4
        ) { patch in
            predFeatSeq.append(patch)
        }

        let B = audioFeat.dim(0)
        let D = config.audioVaeConfig.latentDim
        let predFeatSeqTensor = MLX.concatenated(predFeatSeq, axis: 1) // [B, T, P, D]
        let featPred = predFeatSeqTensor.reshaped([B, D, -1]) // [B, D, T*P]
        let generatedFeat = predFeatSeqTensor[0..., 0..., 0..., 0...].squeezed(axis: 0) // [T, P, D]
        return InferenceResult(featPred: featPred, generatedFeat: generatedFeat, contextLen: 0)
    }

    private func inferenceStream(
        textToken: MLXArray,
        textMask: MLXArray,
        audioFeat: MLXArray,
        audioMask: MLXArray,
        maxLen: Int,
        inferenceTimesteps: Int,
        cfgValue: Float,
        streamingPrefixLen: Int,
        onPatch: (MLXArray) -> Void
    ) throws {
        let B = audioFeat.dim(0)
        let P = config.patchSize

        // Prefill encoder
        let featEmbed = featEncoder(audioFeat)
        let featProj = encToLMProj(featEmbed)

        let scaleEmb = config.lmConfig.useMup ? config.lmConfig.scaleEmb : 1.0
        let textEmbed = baseLM.getEmbeddings(for: textToken) * scaleEmb
        let combinedEmbed = textMask.expandedDimensions(axis: -1) * textEmbed
            + audioMask.expandedDimensions(axis: -1) * featProj

        var prefixFeatCond = audioFeat[0..., -1, 0...]

        var baseCache = (0..<config.lmConfig.numHiddenLayers).map { _ in KVCacheSimple() }
        var residualCache = (0..<config.residualLmNumLayers).map { _ in KVCacheSimple() }

        var encOutputs = baseLM.forwardWithEmbeddings(
            inputsEmbeds: combinedEmbed,
            cache: baseCache,
            mask: .causal
        )
        encOutputs = fsqLayer(encOutputs) * audioMask.expandedDimensions(axis: -1)
            + encOutputs * textMask.expandedDimensions(axis: -1)
        var lmHidden = encOutputs[0..., -1, 0...]

        let residualInputs = fusionConcatProj(
            MLX.concatenated([encOutputs, audioMask.expandedDimensions(axis: -1) * featProj], axis: -1)
        )
        let residualOutputs = residualLM.forwardWithEmbeddings(
            inputsEmbeds: residualInputs,
            cache: residualCache,
            mask: .causal
        )
        var residualHidden = residualOutputs[0..., -1, 0...]

        for i in 0..<maxLen {
            if i % 5 == 0 {
                print("[VoxCPM2] Generating patch \(i)/\(maxLen)...")
            }
            let ditHidden = lmToDitProj(lmHidden) + resToDitProj(residualHidden)

            let predFeat = featDecoder.generate(
                mu: ditHidden,
                nTimesteps: inferenceTimesteps,
                patchSize: P,
                cond: swappedAxes(prefixFeatCond, 1, 2),
                cfgValue: cfgValue,
                temperature: 1.0,
                swaySamplingCoef: 1.0
            ).swappedAxes(1, 2) // [B, P, D]

            let currEmbed = encToLMProj(featEncoder(predFeat.expandedDimensions(axis: 1)))

            onPatch(predFeat.expandedDimensions(axis: 1)) // [B, 1, P, D]
            prefixFeatCond = predFeat

            let baseOut = baseLM.forwardWithEmbeddings(
                inputsEmbeds: currEmbed,
                cache: baseCache,
                mask: .none
            )
            let lmHiddenBeforeFSQ = baseOut.squeezed(axis: 1)

            let stopLogits = stopHead(stopProj(lmHiddenBeforeFSQ))
            let stopFlag = MLX.argMax(stopLogits, axis: -1).item(Int.self)
            print("[VoxCPM2] stopLogits shape: \(stopLogits.shape), stopFlag: \(stopFlag)")
            if stopFlag == 1 {
                print("[VoxCPM2] Stop predictor triggered at patch \(i)")
                break
            }

            lmHidden = fsqLayer(lmHiddenBeforeFSQ)

            let residualInput = fusionConcatProj(
                MLX.concatenated([lmHidden.expandedDimensions(axis: 1), currEmbed], axis: -1)
            )
            let residualOut = residualLM.forwardWithEmbeddings(
                inputsEmbeds: residualInput,
                cache: residualCache,
                mask: .none
            )
            residualHidden = residualOut.squeezed(axis: 1)
        }
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        _ repoID: String,
        cache: HubCache = .default
    ) async throws -> VoxCPM2Model {
        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(repoID)")
        }

        // Download and parse config
        let configURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["config.json"]
        ).appendingPathComponent("config.json")

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(VoxCPM2Config.self, from: configData)

        let model = try VoxCPM2Model(config: config)

        // Load weights
        let weightsURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["model.safetensors"]
        ).appendingPathComponent("model.safetensors")

        let weights = try MLX.loadArrays(url: weightsURL)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Apply per-layer quantization if configured (only for base_lm and residual_lm)
        if let perLayerQuantization = config.perLayerQuantization {
            quantize(model: model) { path, module in
                guard weights["\(path).scales"] != nil else { return nil }
                return perLayerQuantization.quantization(layer: path)?.asTuple
            }
        } else if let quantization = config.quantization {
            let baseLMFilter: (String, Module) -> Bool = { path, _ in
                return path.hasPrefix("base_lm") || path.hasPrefix("residual_lm")
            }
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits, filter: baseLMFilter)
        }

        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .noUnusedKeys)
        eval(model)

        // Load text tokenizer
        model.tokenizer = try await AutoTokenizer.from(modelFolder: {
            let dir = try await ModelUtils.resolveOrDownloadModel(
                repoID: repo,
                requiredExtension: "json",
                additionalMatchingPatterns: ["tokenizer.json"]
            )
            return dir
        }())

        return model
    }

    // MARK: - Weight Sanitization

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            // RoPE frequencies are computed from config, not loaded from weights
            if key.contains(".rope.inv_freq") || key.contains(".rope.long_factor") || key.contains(".rope.short_factor") {
                continue
            }
            // _sr_boundaries is a non-learnable buffer in the decoder
            if key.hasSuffix("_sr_boundaries") {
                continue
            }
            sanitized[key] = value
        }
        return sanitized
    }
}
