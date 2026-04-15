import Foundation
@preconcurrency import MLX
import MLXNN
import MLXFast

// MARK: - Snake Activation

private func snakeActivation(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let a = alpha + 1e-9
    return x + (1.0 / a) * MLX.pow(MLX.sin(a * x), 2)
}

private final class Snake1d: Module, UnaryLayer {
    let alpha: MLXArray

    init(channels: Int) {
        // Checkpoint stores alpha as (1, 1, C); snake operates on NLC internally.
        self.alpha = MLXArray.ones([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = snakeActivation(swappedAxes(x, 1, 2), alpha: alpha)
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - NCL Convolutions (MLXNN Conv1d expects NLC; VoxCPM2 uses NCL like PyTorch)

private class NCLConv1d: Conv1d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(swappedAxes(x, 1, 2))
        return swappedAxes(y, 1, 2)
    }
}

private class NCLConvTransposed1d: ConvTransposed1d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(swappedAxes(x, 1, 2))
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - Causal Convolutions

private final class CausalConv1dLayer: NCLConv1d {
    let leftPad: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.leftPad = padding * 2 - outputPadding
        super.init(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xPadded = x
        if leftPad > 0 {
            let padWidths: [IntOrPair] = [0, 0, IntOrPair((leftPad, 0))]
            xPadded = padded(xPadded, widths: padWidths)
        }
        return super.callAsFunction(xPadded)
    }
}

private final class CausalConvTranspose1dLayer: NCLConvTransposed1d {
    let leftTrim: Int

    override init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.leftTrim = padding * 2 - outputPadding
        super.init(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            outputPadding: outputPadding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        if leftTrim > 0 {
            return y[.ellipsis, 0 ..< (y.dim(-1) - leftTrim)]
        }
        return y
    }
}

// MARK: - Residual Unit

private final class CausalResidualUnit: Module, UnaryLayer {
    let dim: Int
    @ModuleInfo(key: "conv1") var conv1: CausalConv1dLayer
    @ModuleInfo(key: "snake1") var snake1: Snake1d
    @ModuleInfo(key: "conv2") var conv2: CausalConv1dLayer
    @ModuleInfo(key: "snake2") var snake2: Snake1d

    init(dim: Int = 16, dilation: Int = 1, kernelSize: Int = 7, groups: Int = 1) {
        self.dim = dim
        let pad = ((kernelSize - 1) * dilation) / 2
        self._conv1.wrappedValue = CausalConv1dLayer(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: kernelSize,
            padding: pad,
            dilation: dilation,
            groups: groups
        )
        self._snake1.wrappedValue = Snake1d(channels: dim)
        self._conv2.wrappedValue = CausalConv1dLayer(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 1
        )
        self._snake2.wrappedValue = Snake1d(channels: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = snake1(x)
        y = conv1(y)
        y = snake2(y)
        y = conv2(y)
        return x + y
    }
}

// MARK: - Encoder / Decoder Blocks

private final class CausalEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "snake") var snake: Snake1d
    @ModuleInfo(key: "conv") var conv: CausalConv1dLayer
    @ModuleInfo(key: "res1") var res1: CausalResidualUnit
    @ModuleInfo(key: "res2") var res2: CausalResidualUnit
    @ModuleInfo(key: "res3") var res3: CausalResidualUnit

    init(outputDim: Int, inputDim: Int? = nil, stride: Int = 1, groups: Int = 1) {
        let inDim = inputDim ?? (outputDim / 2)
        self._snake.wrappedValue = Snake1d(channels: inDim)
        self._conv.wrappedValue = CausalConv1dLayer(
            inputChannels: inDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: (stride + 1) / 2,
            outputPadding: stride % 2
        )
        self._res1.wrappedValue = CausalResidualUnit(dim: inDim, dilation: 1, groups: groups)
        self._res2.wrappedValue = CausalResidualUnit(dim: inDim, dilation: 3, groups: groups)
        self._res3.wrappedValue = CausalResidualUnit(dim: inDim, dilation: 9, groups: groups)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = res1(h)
        h = res2(h)
        h = res3(h)
        h = conv(h)
        return h
    }
}

private final class CausalDecoderBlock: Module, UnaryLayer {
    let inputChannels: Int
    @ModuleInfo(key: "snake") var snake: Snake1d
    @ModuleInfo(key: "conv_t") var convT: CausalConvTranspose1dLayer
    @ModuleInfo(key: "res1") var res1: CausalResidualUnit
    @ModuleInfo(key: "res2") var res2: CausalResidualUnit
    @ModuleInfo(key: "res3") var res3: CausalResidualUnit

    init(
        inputDim: Int = 16,
        outputDim: Int = 8,
        stride: Int = 1,
        groups: Int = 1,
        useNoiseBlock: Bool = false
    ) {
        self.inputChannels = inputDim
        self._snake.wrappedValue = Snake1d(channels: inputDim)
        self._convT.wrappedValue = CausalConvTranspose1dLayer(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: (stride + 1) / 2,
            outputPadding: stride % 2
        )
        self._res1.wrappedValue = CausalResidualUnit(dim: outputDim, dilation: 1, groups: groups)
        self._res2.wrappedValue = CausalResidualUnit(dim: outputDim, dilation: 3, groups: groups)
        self._res3.wrappedValue = CausalResidualUnit(dim: outputDim, dilation: 9, groups: groups)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = convT(h)
        h = res1(h)
        h = res2(h)
        h = res3(h)
        return h
    }
}

private final class CausalEncoderBlocks: Module {
    @ModuleInfo(key: "layers") var layers: [CausalEncoderBlock]

    init(layers: [CausalEncoderBlock]) {
        self._layers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

private final class CausalDecoderBlocks: Module {
    @ModuleInfo(key: "layers") var layers: [CausalDecoderBlock]

    init(layers: [CausalDecoderBlock]) {
        self._layers.wrappedValue = layers
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

// MARK: - Sample Rate Condition Layer

private final class SampleRateConditionLayer: Module {
    let condType: String
    @ModuleInfo(key: "scale_embed") var scaleEmbed: Embedding?
    @ModuleInfo(key: "bias_embed") var biasEmbed: Embedding?
    @ModuleInfo(key: "cond_embed") var condEmbed: Embedding?
    @ModuleInfo(key: "out_layer") var outLayer: NCLConv1d?

    init(
        inputDim: Int,
        srBinBuckets: Int,
        condType: String = "scale_bias",
        condDim: Int = 128,
        outLayer: Bool = false
    ) {
        self.condType = condType

        switch condType {
        case "scale_bias", "scale_bias_init":
            self._scaleEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
            self._biasEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
            self._condEmbed.wrappedValue = nil
        case "add":
            self._condEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: inputDim)
            self._scaleEmbed.wrappedValue = nil
            self._biasEmbed.wrappedValue = nil
        case "concat":
            self._condEmbed.wrappedValue = Embedding(embeddingCount: srBinBuckets, dimensions: condDim)
            self._scaleEmbed.wrappedValue = nil
            self._biasEmbed.wrappedValue = nil
        default:
            fatalError("Invalid cond_type: \(condType)")
        }

        if outLayer {
            self._outLayer.wrappedValue = NCLConv1d(
                inputChannels: condType == "concat" ? inputDim + condDim : inputDim,
                outputChannels: inputDim,
                kernelSize: 1
            )
        } else {
            self._outLayer.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, srCond: MLXArray) -> MLXArray {
        var y = x
        switch condType {
        case "scale_bias", "scale_bias_init":
            let scale = scaleEmbed!(srCond).reshaped([1, -1, 1])
            let bias = biasEmbed!(srCond).reshaped([1, -1, 1])
            y = y * scale + bias
        case "add":
            let emb = condEmbed!(srCond).reshaped([1, -1, 1])
            y = y + emb
        case "concat":
            let emb = condEmbed!(srCond).reshaped([1, -1, 1])
            let embTiled = MLX.broadcast(emb, to: [1, emb.dim(1), x.dim(-1)])
            y = MLX.concatenated([y, embTiled], axis: 1)
        default:
            break
        }
        if let outLayer {
            y = outLayer(y)
        }
        return y
    }
}

// MARK: - Encoder

public final class CausalEncoder: Module {
    public let latentDim: Int
    public let encDim: Int

    @ModuleInfo(key: "conv_in") fileprivate var convIn: CausalConv1dLayer
    @ModuleInfo(key: "blocks") fileprivate var blocks: CausalEncoderBlocks
    @ModuleInfo(key: "fc_mu") fileprivate var fcMu: NCLConv1d

    public init(
        dModel: Int = 64,
        latentDim: Int = 32,
        strides: [Int] = [2, 5, 8, 8],
        depthwise: Bool = false
    ) {
        self.latentDim = latentDim
        self._convIn.wrappedValue = CausalConv1dLayer(
            inputChannels: 1,
            outputChannels: dModel,
            kernelSize: 7,
            padding: 3
        )

        var blockList: [CausalEncoderBlock] = []
        var currentDim = dModel
        for stride in strides {
            let outDim = currentDim * 2
            let groups = depthwise ? (currentDim / 2) : 1
            blockList.append(CausalEncoderBlock(
                outputDim: outDim,
                inputDim: currentDim,
                stride: stride,
                groups: groups
            ))
            currentDim = outDim
        }
        self._blocks.wrappedValue = CausalEncoderBlocks(layers: blockList)
        self.encDim = currentDim

        self._fcMu.wrappedValue = NCLConv1d(
            inputChannels: currentDim,
            outputChannels: latentDim,
            kernelSize: 3,
            padding: 1,
            groups: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        h = blocks(h)
        let y = snakeActivation(swappedAxes(fcMu(h), 1, 2), alpha: MLXArray.ones([1, 1, latentDim]))
        return swappedAxes(y, 1, 2)
    }
}

// MARK: - Decoder

public final class CausalDecoder: Module {
    public let srBinBoundaries: [Int]?
    public let srBinBuckets: Int
    public let condType: String
    public let condDim: Int
    public let condOutLayer: Bool

    @ModuleInfo(key: "conv_in") fileprivate var convIn: Sequential
    @ModuleInfo(key: "blocks") fileprivate var blocks: CausalDecoderBlocks
    @ModuleInfo(key: "snake_out") fileprivate var snakeOut: Snake1d
    @ModuleInfo(key: "conv_out") fileprivate var convOut: NCLConv1d
    @ModuleInfo(key: "sr_cond_layers") fileprivate var srCondLayers: [SampleRateConditionLayer?]

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        depthwise: Bool = false,
        dOut: Int = 1,
        useNoiseBlock: Bool = false,
        srBinBoundaries: [Int]? = nil,
        condType: String = "scale_bias",
        condDim: Int = 128,
        condOutLayer: Bool = false
    ) {
        self.srBinBoundaries = srBinBoundaries
        self.srBinBuckets = (srBinBoundaries?.count ?? 0) + 1
        self.condType = condType
        self.condDim = condDim
        self.condOutLayer = condOutLayer

        var convInLayers: [any UnaryLayer] = []
        if depthwise {
            convInLayers.append(CausalConv1dLayer(
                inputChannels: inputChannel,
                outputChannels: inputChannel,
                kernelSize: 7,
                padding: 3,
                groups: inputChannel
            ) as any UnaryLayer)
            convInLayers.append(NCLConv1d(
                inputChannels: inputChannel,
                outputChannels: channels,
                kernelSize: 1
            ) as any UnaryLayer)
        } else {
            convInLayers.append(CausalConv1dLayer(
                inputChannels: inputChannel,
                outputChannels: channels,
                kernelSize: 7,
                padding: 3
            ) as any UnaryLayer)
        }
        self._convIn.wrappedValue = Sequential(layers: convInLayers)

        var blockList: [CausalDecoderBlock] = []
        for (i, stride) in rates.enumerated() {
            let inputDim = channels / (1 << i)
            let outputDim = channels / (1 << (i + 1))
            let groups = depthwise ? outputDim : 1
            blockList.append(CausalDecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride,
                groups: groups,
                useNoiseBlock: useNoiseBlock
            ))
        }
        self._blocks.wrappedValue = CausalDecoderBlocks(layers: blockList)

        let finalOutputDim = channels / (1 << rates.count)
        self._snakeOut.wrappedValue = Snake1d(channels: finalOutputDim)
        self._convOut.wrappedValue = NCLConv1d(
            inputChannels: finalOutputDim,
            outputChannels: dOut,
            kernelSize: 7,
            padding: 3
        )

        var condLayers: [SampleRateConditionLayer?] = []
        if srBinBoundaries != nil {
            for block in blockList {
                condLayers.append(SampleRateConditionLayer(
                    inputDim: block.inputChannels,
                    srBinBuckets: self.srBinBuckets,
                    condType: condType,
                    condDim: condDim,
                    outLayer: condOutLayer
                ))
            }
        }
        self._srCondLayers.wrappedValue = condLayers
    }

    private func getSRIdx(sr: Int) -> Int {
        guard let boundaries = srBinBoundaries else { return 0 }
        var idx = 0
        for boundary in boundaries {
            if sr > boundary {
                idx += 1
            } else {
                break
            }
        }
        return min(idx, srBinBuckets - 1)
    }

    public func callAsFunction(_ x: MLXArray, srCond: Int? = nil) -> MLXArray {
        var h = convIn(x)

        if srBinBoundaries != nil {
            let srIdx = MLXArray(getSRIdx(sr: srCond ?? 48000))
            for (i, block) in blocks.layers.enumerated() {
                if let condLayer = srCondLayers[i] {
                    h = condLayer(h, srCond: srIdx)
                }
                h = block(h)
            }
        } else {
            h = blocks(h)
        }

        h = snakeOut(h)
        h = convOut(h)
        return MLX.tanh(h)
    }
}

// MARK: - Audio VAE V2

public final class VoxCPM2AudioVAE: Module {
    public let config: VoxCPM2AudioVAEConfig
    public let latentDim: Int
    public let hopLength: Int
    public let chunkSize: Int
    public let decodeChunkSize: Int
    public let sampleRate: Int
    public let outSampleRate: Int

    @ModuleInfo(key: "encoder") public var encoder: CausalEncoder
    @ModuleInfo(key: "decoder") public var decoder: CausalDecoder

    public init(config: VoxCPM2AudioVAEConfig, depthwise: Bool = true, useNoiseBlock: Bool = false) {
        self.config = config
        self.sampleRate = config.sampleRate
        self.outSampleRate = config.outSampleRate
        self.latentDim = config.latentDim
        self.hopLength = config.encoderRates.reduce(1, *)
        self.chunkSize = self.hopLength
        self.decodeChunkSize = config.decoderRates.reduce(1, *)

        self._encoder.wrappedValue = CausalEncoder(
            dModel: config.encoderDim,
            latentDim: config.latentDim,
            strides: config.encoderRates,
            depthwise: depthwise
        )

        self._decoder.wrappedValue = CausalDecoder(
            inputChannel: config.latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates,
            depthwise: depthwise,
            dOut: 1,
            useNoiseBlock: useNoiseBlock,
            srBinBoundaries: config.srBinBoundaries,
            condType: "scale_bias",
            condDim: 128,
            condOutLayer: false
        )
    }

    /// Encode audio waveform to latent.
    /// - Parameter audio: [B, 1, T] or [B, T]
    /// - Returns: [B, latentDim, T']
    public func encode(_ audio: MLXArray) -> MLXArray {
        var x = audio
        if x.ndim == 2 {
            x = x.expandedDimensions(axis: 1)
        }
        x = preprocess(x)
        return encoder(x)
    }

    /// Decode latent to audio waveform.
    /// - Parameter z: [B, latentDim, T']
    /// - Returns: [B, 1, T]
    public func decode(_ z: MLXArray, srCond: Int? = nil) -> MLXArray {
        decoder(z, srCond: srCond)
    }

    private func preprocess(_ audio: MLXArray) -> MLXArray {
        let length = audio.dim(-1)
        let padTo = hopLength
        let rightPad = ((length + padTo - 1) / padTo) * padTo - length
        if rightPad > 0 {
            let padWidths: [IntOrPair] = [0, 0, IntOrPair((0, rightPad))]
            return padded(audio, widths: padWidths)
        }
        return audio
    }
}
