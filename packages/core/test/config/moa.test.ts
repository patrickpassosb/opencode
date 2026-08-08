import { describe, expect, it } from "bun:test"
import { Exit, Schema } from "effect"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"

const validMoaConfig = {
  moa: {
    default_preset: "glm5.2+minimaxm3+deepseekv4flash0731",
    save_traces: true,
    presets: {
      "glm5.2+minimaxm3+deepseekv4flash0731": {
        advisors: [
          { provider: "custom", model: "ollama-cloud/glm-5.2", maxTokens: 600 },
          { provider: "custom", model: "ollama-cloud/minimax-m3" },
        ],
        aggregator: { provider: "custom", model: "ollama-cloud/deepseek-v4-flash:0731", maxTokens: 4096 },
        maxTokens: 4096,
        referenceMaxTokens: 600,
        fanout: "user_turn",
        temperature: 0.7,
      },
    },
  },
}

function decode(input: unknown) {
  return Schema.decodeUnknownExit(ConfigV1.Info)(input, { errors: "all", propertyOrder: "original" })
}

describe("MoA config schema", () => {
  it("decodes a full moa preset config", () => {
    const exit = decode(validMoaConfig)
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      const moa = exit.value.moa
      expect(moa).toBeDefined()
      expect(moa!.default_preset).toBe("glm5.2+minimaxm3+deepseekv4flash0731")
      expect(moa!.save_traces).toBe(true)
      const preset = moa!.presets["glm5.2+minimaxm3+deepseekv4flash0731"]
      expect(preset.advisors).toHaveLength(2)
      expect(preset.advisors[0].provider).toBe("custom")
      expect(preset.advisors[0].maxTokens).toBe(600)
      expect(preset.advisors[1].maxTokens).toBeUndefined()
      expect(preset.aggregator.model).toBe("ollama-cloud/deepseek-v4-flash:0731")
      expect(preset.aggregator.maxTokens).toBe(4096)
      expect(preset.fanout).toBe("user_turn")
      expect(preset.temperature).toBe(0.7)
    }
  })

  it("omits moa entirely when not configured", () => {
    const exit = decode({ model: "custom/some-model" })
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect(exit.value.moa).toBeUndefined()
    }
  })

  it("rejects a preset with a non-array advisors field", () => {
    const exit = decode({
      moa: {
        presets: {
          broken: {
            advisors: "not-an-array",
            aggregator: { provider: "custom", model: "aggregator" },
          },
        },
      },
    })
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("rejects a preset missing the aggregator", () => {
    const exit = decode({
      moa: {
        presets: {
          broken: {
            advisors: [{ provider: "custom", model: "advisor" }],
          },
        },
      },
    })
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("rejects an invalid fanout mode", () => {
    const exit = decode({
      moa: {
        presets: {
          broken: {
            advisors: [{ provider: "custom", model: "advisor" }],
            aggregator: { provider: "custom", model: "aggregator" },
            fanout: "per-message",
          },
        },
      },
    })
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("rejects a preset with an invalid advisor entry (missing model)", () => {
    const exit = decode({
      moa: {
        presets: {
          broken: {
            advisors: [{ provider: "custom" }],
            aggregator: { provider: "custom", model: "aggregator" },
          },
        },
      },
    })
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("strips unknown nested preset keys per config convention", () => {
    // ConfigV1 nested structs are non-exact by repo convention (top-level keys
    // are the strict boundary via ConfigParse). Unknown preset keys are dropped.
    const exit = decode({
      moa: {
        presets: {
          ok: {
            advisors: [{ provider: "custom", model: "advisor" }],
            aggregator: { provider: "custom", model: "aggregator" },
            future_key: 42,
          },
        },
      },
    })
    expect(Exit.isSuccess(exit)).toBe(true)
    if (Exit.isSuccess(exit)) {
      expect("future_key" in exit.value.moa!.presets.ok).toBe(false)
    }
  })
})
