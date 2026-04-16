import Testing
import MLX
import MLXNN
import MLXFast
import Foundation
import HuggingFace

@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite("VoxCPM2")
struct VoxCPM2Tests {

    @Test func configDecodesFromJSON() throws {
        let json = """
        {
            "model_type": "voxcpm2",
            "architecture": "voxcpm2",
            "lm_config": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "max_position_embeddings": 32768,
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-05,
                "rope_theta": 10000,
                "kv_channels": 128,
                "rope_scaling": {
                    "type": "longrope",
                    "long_factor": [1.0, 1.0],
                    "short_factor": [1.0, 1.0],
                    "original_max_position_embeddings": 32768
                },
                "vocab_size": 73448,
                "use_mup": false,
                "scale_emb": 12,
                "dim_model_base": 256,
                "scale_depth": 1.4
            },
            "patch_size": 4,
            "feat_dim": 64,
            "scalar_quantization_latent_dim": 512,
            "scalar_quantization_scale": 9,
            "residual_lm_num_layers": 8,
            "residual_lm_no_rope": true,
            "encoder_config": {
                "hidden_dim": 1024,
                "ffn_dim": 4096,
                "num_heads": 16,
                "num_layers": 12,
                "kv_channels": 128
            },
            "dit_config": {
                "hidden_dim": 1024,
                "ffn_dim": 4096,
                "num_heads": 16,
                "num_layers": 12,
                "kv_channels": 128,
                "mean_mode": false,
                "cfm_config": {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "log-norm",
                    "inference_cfg_rate": 2.0
                }
            },
            "audio_vae_config": {
                "encoder_dim": 128,
                "encoder_rates": [2, 5, 8, 8],
                "latent_dim": 64,
                "decoder_dim": 2048,
                "decoder_rates": [8, 6, 5, 2, 2, 2],
                "sr_bin_boundaries": [20000, 30000, 40000],
                "sample_rate": 16000,
                "out_sample_rate": 48000
            },
            "max_length": 8192,
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "targets": ["base_lm", "residual_lm"]
            }
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(VoxCPM2Config.self, from: json)
        #expect(config.modelType == "voxcpm2")
        #expect(config.lmConfig.hiddenSize == 2048)
        #expect(config.lmConfig.numHiddenLayers == 28)
        #expect(config.residualLmNumLayers == 8)
        #expect(config.encoderConfig.hiddenDim == 1024)
        #expect(config.ditConfig.hiddenDim == 1024)
        #expect(config.audioVaeConfig.latentDim == 64)
        #expect(config.audioVaeConfig.outSampleRate == 48000)
        #expect(config.quantization?.bits == 4)
        #expect(config.quantization?.targets == ["base_lm", "residual_lm"])
    }

    @Test func modelInstantiatesFromConfig() throws {
        let config = try makeTinyVoxCPM2Config()
        let model = try VoxCPM2Model(config: config)
        #expect(model.sampleRate == 48000)
    }

    @Test func modelStructureMatchesWeightKeys() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["MLXAUDIO_TEST_MODEL_DIR"] else {
            print("⚠️ Skipping: set MLXAUDIO_TEST_MODEL_DIR to model directory")
            return
        }
        let modelDir = URL(fileURLWithPath: dirPath)
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("⚠️ Skipping: config.json not found at \(configURL.path)")
            return
        }

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(VoxCPM2Config.self, from: configData)
        let model = try VoxCPM2Model(config: config)

        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let sanitized = model.sanitize(weights: rawWeights)

        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let weightKeys = Set(sanitized.keys)

        let missingInModel = weightKeys.subtracting(modelKeys)
        let missingInWeights = modelKeys.subtracting(weightKeys)

        if !missingInModel.isEmpty {
            print("❌ Weight keys not found in model (\(missingInModel.count)):")
            for k in missingInModel.sorted().prefix(30) { print("  \(k)") }
        }
        if !missingInWeights.isEmpty {
            print("⚠️ Model keys not in weights (\(missingInWeights.count)):")
            for k in missingInWeights.sorted().prefix(30) { print("  \(k)") }
        }

        #expect(missingInModel.count == 0, "Weight keys not matched by model structure")
    }

    @Suite("VoxCPM2 Network Tests", .serialized)
    struct VoxCPM2NetworkTests {
        @Test func downloadAndValidateWeightMapping() async throws {
            let env = ProcessInfo.processInfo.environment
            guard env["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1" else {
                print("Skipping network VoxCPM2 test. Set MLXAUDIO_ENABLE_NETWORK_TESTS=1 to enable.")
                return
            }

            let repo = env["MLXAUDIO_VOXCPM2_REPO"] ?? "mlx-community/VoxCPM2-4bit"
            let model = try await VoxCPM2Model.fromPretrained(repo)

            // Model keys after loading (quantization may change some layer types)
            let modelKeys = Set(model.parameters().flattened().map(\.0))
            print("Loaded model keys: \(modelKeys.count)")

            // Re-load raw weights to inspect original keys
            guard let repoID = Repo.ID(rawValue: repo) else {
                throw AudioGenerationError.invalidInput("Invalid repository ID: \(repo)")
            }
            let weightsURL = try await ModelUtils.resolveOrDownloadModel(
                repoID: repoID,
                requiredExtension: ".safetensors",
                additionalMatchingPatterns: ["model.safetensors"]
            ).appendingPathComponent("model.safetensors")
            let rawWeights = try MLX.loadArrays(url: weightsURL)
            let weightKeys = Set(rawWeights.keys)

            print("Raw weight keys: \(weightKeys.count)")

            let sanitized = model.sanitize(weights: rawWeights)
            let sanitizedKeys = Set(sanitized.keys)
            let missingInModel = sanitizedKeys.subtracting(modelKeys)
            let missingInWeights = modelKeys.subtracting(sanitizedKeys)

            if !missingInModel.isEmpty {
                print("❌ Weight keys not found in model (\(missingInModel.count)):")
                for k in missingInModel.sorted().prefix(40) { print("  \(k)") }
            }
            if !missingInWeights.isEmpty {
                print("⚠️ Model keys not in weights (\(missingInWeights.count)):")
                for k in missingInWeights.sorted().prefix(40) { print("  \(k)") }
            }

            #expect(missingInModel.count == 0, "Weight keys not matched by model structure")
        }
    }

    @Test func factoryInfersVoxCPM2ModelType() {
        let resolved = TTS.resolveModelType(modelRepo: "mlx-community/VoxCPM2-4bit")
        #expect(resolved == "voxcpm2")
    }
}

private func makeTinyVoxCPM2Config() throws -> VoxCPM2Config {
    let json = """
    {
        "model_type": "voxcpm2",
        "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000,
            "vocab_size": 128,
            "use_mup": false,
            "scale_emb": 1.0,
            "dim_model_base": 16,
            "scale_depth": 1.0
        },
        "patch_size": 4,
        "feat_dim": 8,
        "scalar_quantization_latent_dim": 16,
        "scalar_quantization_scale": 1,
        "residual_lm_num_layers": 1,
        "residual_lm_no_rope": true,
        "encoder_config": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "num_heads": 2,
            "num_layers": 1
        },
        "dit_config": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "mean_mode": false,
            "cfm_config": {
                "sigma_min": 1e-06,
                "solver": "euler",
                "t_scheduler": "log-norm",
                "inference_cfg_rate": 2.0
            }
        },
        "audio_vae_config": {
            "encoder_dim": 16,
            "encoder_rates": [2, 2],
            "latent_dim": 8,
            "decoder_dim": 64,
            "decoder_rates": [2, 2],
            "sample_rate": 16000,
            "out_sample_rate": 48000
        },
        "max_length": 128
    }
    """.data(using: .utf8)!
    return try JSONDecoder().decode(VoxCPM2Config.self, from: json)
}
