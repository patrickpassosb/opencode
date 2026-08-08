import { describe, expect, it } from "bun:test"
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider"
import type { ConfigMoAV1 } from "@opencode-ai/core/v1/config/moa"
import { moaLanguageModel } from "../../src/provider/moa"

type CallLog = Array<{ model: string; prompt: LanguageModelV3Prompt; maxOutputTokens?: number }>

function fakeModel(modelID: string, reply: string, log: CallLog, fail = false): LanguageModelV3 {
  return {
    specificationVersion: "v3",
    provider: "fake",
    modelId: modelID,
    supportedUrls: {},
    async doGenerate(options) {
      log.push({ model: modelID, prompt: options.prompt, maxOutputTokens: options.maxOutputTokens })
      if (fail) throw new Error(`boom from ${modelID}`)
      if (options.abortSignal?.aborted) throw options.abortSignal.reason ?? new Error("aborted")
      return {
        content: [{ type: "text", text: reply }],
        finishReason: { unified: "stop", raw: "stop" },
        usage: {
          inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 5, text: 5, reasoning: 0 },
        },
        warnings: [],
      }
    },
    async doStream(options) {
      log.push({ model: modelID, prompt: options.prompt, maxOutputTokens: options.maxOutputTokens })
      if (fail) throw new Error(`boom from ${modelID}`)
      const encoder = new TextEncoder()
      const chunks: LanguageModelV3StreamPart[] = [
        { type: "text-start", id: "1" },
        { type: "text-delta", id: "1", delta: reply },
        { type: "text-end", id: "1" },
        {
          type: "finish",
          finishReason: { unified: "stop", raw: "stop" },
          usage: {
            inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 5, text: 5, reasoning: 0 },
          },
        },
      ]
      return {
        stream: new ReadableStream<LanguageModelV3StreamPart>({
          start(controller) {
            for (const chunk of chunks) controller.enqueue(chunk)
            controller.close()
          },
        }),
      }
    },
  }
}

const preset: ConfigMoAV1.Preset = {
  advisors: [
    { provider: "fake", model: "advisor-a" },
    { provider: "fake", model: "advisor-b" },
  ],
  aggregator: { provider: "fake", model: "aggregator" },
  maxTokens: 4096,
  referenceMaxTokens: 600,
  fanout: "user_turn",
}

const options: LanguageModelV3CallOptions = {
  prompt: [
    { role: "system", content: "sys" },
    { role: "user", content: [{ type: "text", text: "hello" }] },
  ],
}

function makeEngine(log: CallLog, failures: Record<string, boolean> = {}) {
  const models = new Map<string, LanguageModelV3>()
  models.set("advisor-a", fakeModel("advisor-a", "analysis-a", log, failures["advisor-a"] ?? false))
  models.set("advisor-b", fakeModel("advisor-b", "analysis-b", log, failures["advisor-b"] ?? false))
  models.set("aggregator", fakeModel("aggregator", "final answer", log, failures["aggregator"] ?? false))
  return moaLanguageModel(preset, async (providerID, modelID) => {
    const model = models.get(modelID)
    if (!model) throw new Error(`unknown model ${providerID}/${modelID}`)
    return model
  })
}

describe("MoA fan-out engine", () => {
  it("fans out to all advisors in parallel and aggregates the final answer", async () => {
    const log: CallLog = []
    const engine = makeEngine(log)

    const result = await engine.doGenerate(options)

    expect(result.content).toEqual([{ type: "text", text: "final answer" }])
    const advisorCalls = log.filter((entry) => entry.model.startsWith("advisor"))
    expect(advisorCalls).toHaveLength(2)
    for (const call of advisorCalls) expect(call.maxOutputTokens).toBe(600)
    expect(advisorCalls.every((call) => call.prompt === options.prompt)).toBe(true)

    const aggregatorCall = log.find((entry) => entry.model === "aggregator")
    expect(aggregatorCall).toBeDefined()
    expect(aggregatorCall!.maxOutputTokens).toBe(4096)
    const prompt = aggregatorCall!.prompt
    const advisorTexts = prompt.filter((message) => message.role === "assistant")
    expect(advisorTexts.some((message) => message.content[0].type === "text" && message.content[0].text.includes("analysis-a"))).toBe(true)
    expect(advisorTexts.some((message) => message.content[0].type === "text" && message.content[0].text.includes("analysis-b"))).toBe(true)
  })

  it("isolates a failing advisor and still produces a final answer", async () => {
    const log: CallLog = []
    const engine = makeEngine(log, { "advisor-b": true })

    const result = await engine.doGenerate(options)

    expect(result.content).toEqual([{ type: "text", text: "final answer" }])
    const usage = engine.moaUsage()
    expect(usage.advisors[0].status).toBe("ok")
    expect(usage.advisors[1].status).toBe("error")
    expect(usage.advisors[1].error).toContain("boom from advisor-b")
    expect(usage.aggregator.status).toBe("ok")
  })

  it("throws a descriptive error when all advisors fail", async () => {
    const log: CallLog = []
    const engine = makeEngine(log, { "advisor-a": true, "advisor-b": true })

    await expect(engine.doGenerate(options)).rejects.toThrow("all 2 MoA advisors failed")
  })

  it("streams the aggregator output with merged usage on the finish part", async () => {
    const log: CallLog = []
    const engine = makeEngine(log)

    const result = await engine.doStream(options)
    const parts: LanguageModelV3StreamPart[] = []
    const reader = result.stream.getReader()
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      parts.push(value)
    }

    const text = parts
      .filter((part) => part.type === "text-delta")
      .map((part) => (part.type === "text-delta" ? part.delta : ""))
      .join("")
    expect(text).toBe("final answer")

    const finish = parts.find((part) => part.type === "finish")
    expect(finish).toBeDefined()
    if (finish?.type === "finish") {
      // 2 advisors (10 in / 5 out each) + aggregator (10 in / 5 out)
      expect(finish.usage.inputTokens.total).toBe(30)
      expect(finish.usage.outputTokens.total).toBe(15)
    }
    expect(engine.moaUsage().aggregator.status).toBe("ok")
  })

  it("propagates abort signals into advisor calls", async () => {
    const log: CallLog = []
    const engine = makeEngine(log)
    const ctl = new AbortController()
    ctl.abort()

    // Abort takes precedence over the failure-isolation message: an aborted
    // turn surfaces the abort reason instead of a synthetic error.
    await expect(
      engine.doGenerate({ ...options, abortSignal: ctl.signal }),
    ).rejects.toThrow("The operation was aborted.")
  })
})
