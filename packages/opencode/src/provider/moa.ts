export * as MoA from "./moa"

import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider"
import { ConfigMoAV1 } from "@opencode-ai/core/v1/config/moa"
import { mkdir, writeFile } from "fs/promises"
import path from "path"

const ADVISOR_SYSTEM =
  "You are an advisor in a mixture-of-agents ensemble. Analyze the conversation and produce a concise, focused analysis of the problem and the best candidate directions. Do not answer as the final assistant; return only your analysis."

const AGGREGATOR_SYSTEM =
  "You are the aggregator of a mixture-of-agents ensemble. Advisor analyses follow the conversation. Synthesize them into a single final answer that best serves the user."

export type MoAActorStatus = {
  provider: string
  model: string
  status: "pending" | "ok" | "error"
  inputTokens?: number
  outputTokens?: number
  error?: string
}

export type MoAUsage = {
  advisors: MoAActorStatus[]
  aggregator: MoAActorStatus
}

export type MoATrace = {
  preset: string
  startedAt: string
  usage: MoAUsage
}

export type MoAResolver = (providerID: string, modelID: string) => Promise<LanguageModelV3>

export interface MoALanguageModel extends LanguageModelV3 {
  readonly moaUsage: () => MoAUsage
}

function extractText(content: LanguageModelV3GenerateResult["content"]): string {
  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("\n")
}

function mergeUsage(usages: LanguageModelV3Usage[]): LanguageModelV3Usage {
  const total = (pick: (usage: LanguageModelV3Usage) => number | undefined): number | undefined => {
    const sum = usages.reduce((acc, usage) => acc + (pick(usage) ?? 0), 0)
    return sum > 0 ? sum : undefined
  }
  return {
    inputTokens: {
      total: total((usage) => usage.inputTokens.total),
      noCache: total((usage) => usage.inputTokens.noCache),
      cacheRead: total((usage) => usage.inputTokens.cacheRead),
      cacheWrite: total((usage) => usage.inputTokens.cacheWrite),
    },
    outputTokens: {
      total: total((usage) => usage.outputTokens.total),
      text: total((usage) => usage.outputTokens.text),
      reasoning: total((usage) => usage.outputTokens.reasoning),
    },
  }
}

function aggregatorPrompt(prompt: LanguageModelV3Prompt, analyses: Array<{ name: string; text: string }>): LanguageModelV3Prompt {
  return [
    { role: "system", content: AGGREGATOR_SYSTEM },
    ...prompt,
    ...analyses.map((analysis) => ({
      role: "assistant" as const,
      content: [{ type: "text" as const, text: `[advisor ${analysis.name}] ${analysis.text}` }],
    })),
  ]
}

/**
 * Builds the LanguageModelV3 wrapper for a MoA preset. Each advisor receives
 * the full prompt with a capped output; the aggregator synthesizes the final
 * answer from the surviving analyses. A failing advisor never fails the call
 * as long as at least one advisor succeeds.
 */
export function moaLanguageModel(
  preset: ConfigMoAV1.Preset,
  resolve: MoAResolver,
  opts: { presetName?: string; onTrace?: (trace: MoATrace) => void } = {},
): MoALanguageModel {
  const presetName = opts.presetName ?? "moa"
  const advisorCap = preset.referenceMaxTokens ?? 600
  const aggregatorCap = preset.maxTokens ?? 4096

  const usage: MoAUsage = {
    advisors: preset.advisors.map((advisor) => ({
      provider: advisor.provider,
      model: advisor.model,
      status: "pending" as const,
    })),
    aggregator: { provider: preset.aggregator.provider, model: preset.aggregator.model, status: "pending" as const },
  }

  function emitTrace() {
    opts.onTrace?.({ preset: presetName, startedAt: new Date().toISOString(), usage })
  }

  async function runAdvisor(
    advisor: ConfigMoAV1.Advisor,
    options: LanguageModelV3CallOptions,
  ): Promise<{ ok: true; text: string; usage: LanguageModelV3Usage } | { ok: false; error: string }> {
    try {
      const language = await resolve(advisor.provider, advisor.model)
      const result = await language.doGenerate({
        ...options,
        tools: undefined,
        toolChoice: undefined,
        maxOutputTokens: advisor.maxTokens ?? advisorCap,
        temperature: preset.temperature ?? options.temperature,
      })
      return { ok: true, text: extractText(result.content), usage: result.usage }
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) }
    }
  }

  async function collectAnalyses(options: LanguageModelV3CallOptions) {
    const results = await Promise.all(preset.advisors.map((advisor) => runAdvisor(advisor, options)))
    usage.advisors = results.map((result, index) => ({
      provider: preset.advisors[index].provider,
      model: preset.advisors[index].model,
      status: result.ok ? ("ok" as const) : ("error" as const),
      inputTokens: result.ok ? result.usage.inputTokens.total : undefined,
      outputTokens: result.ok ? result.usage.outputTokens.total : undefined,
      error: result.ok ? undefined : result.error,
    }))
    const surviving = results.flatMap((result, index) =>
      result.ok ? [{ name: `advisor-${index}`, text: result.text, usage: result.usage }] : [],
    )
    if (surviving.length === 0) {
      const message = `all ${preset.advisors.length} MoA advisors failed: ${usage.advisors
        .map((advisor) => `${advisor.model}: ${advisor.error}`)
        .join("; ")}`
      if (options.abortSignal?.aborted) throw options.abortSignal.reason
      throw new Error(message)
    }
    return surviving
  }

  return {
    specificationVersion: "v3",
    provider: "moa",
    modelId: presetName,
    supportedUrls: {},
    moaUsage: () => usage,

    async doGenerate(options) {
      const surviving = await collectAnalyses(options)
      const language = await resolve(preset.aggregator.provider, preset.aggregator.model)
      const result = await language.doGenerate({
        ...options,
        prompt: aggregatorPrompt(options.prompt, surviving),
        maxOutputTokens: aggregatorCap,
        temperature: preset.temperature ?? options.temperature,
      })
      usage.aggregator = {
        provider: preset.aggregator.provider,
        model: preset.aggregator.model,
        status: "ok",
        inputTokens: result.usage.inputTokens.total,
        outputTokens: result.usage.outputTokens.total,
      }
      emitTrace()
      return { ...result, usage: mergeUsage([...surviving.map((analysis) => analysis.usage), result.usage]) }
    },

    async doStream(options) {
      const surviving = await collectAnalyses(options)
      const language = await resolve(preset.aggregator.provider, preset.aggregator.model)
      const result = await language.doStream({
        ...options,
        prompt: aggregatorPrompt(options.prompt, surviving),
        maxOutputTokens: aggregatorCap,
        temperature: preset.temperature ?? options.temperature,
      })

      let streamedUsage: LanguageModelV3Usage | undefined
      const stream = result.stream.pipeThrough(
        new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
          transform(part, controller) {
            if (part.type === "finish") {
              streamedUsage = part.usage
              controller.enqueue({ ...part, usage: mergeUsage([...surviving.map((analysis) => analysis.usage), part.usage]) })
            } else {
              controller.enqueue(part)
            }
          },
          flush() {
            if (streamedUsage) {
              usage.aggregator = {
                provider: preset.aggregator.provider,
                model: preset.aggregator.model,
                status: "ok",
                inputTokens: streamedUsage.inputTokens.total,
                outputTokens: streamedUsage.outputTokens.total,
              }
              emitTrace()
            }
          },
        }),
      )
      return { ...result, stream }
    },
  }
}

/**
 * Wraps a MoA language model so each completed turn writes a JSON trace
 * (advisors + aggregator usage) to `dir` when `enabled`. Writes are
 * fire-and-forget; a failed trace write never affects the turn.
 */
export function moaTraceWriter(model: MoALanguageModel, dir: string, enabled: boolean): MoALanguageModel {
  if (!enabled) return model
  const onTrace = (trace: MoATrace) => {
    const file = path.join(dir, `${trace.preset}-${Date.now()}.json`)
    void mkdir(dir, { recursive: true })
      .then(() => writeFile(file, JSON.stringify(trace, null, 2)))
      .catch(() => undefined)
  }
  const traced = moaLanguageModelFrom(model, onTrace)
  return traced
}

function moaLanguageModelFrom(model: MoALanguageModel, onTrace: (trace: MoATrace) => void): MoALanguageModel {
  return {
    specificationVersion: "v3",
    provider: "moa",
    modelId: model.modelId,
    supportedUrls: {},
    moaUsage: () => model.moaUsage(),
    doGenerate: (options) => {
      const inner = model as LanguageModelV3
      return inner.doGenerate(options).then((result) => {
        onTrace({ preset: model.modelId, startedAt: new Date().toISOString(), usage: model.moaUsage() })
        return result
      })
    },
    doStream: (options) => {
      const inner = model as LanguageModelV3
      return inner.doStream(options).then((result) => {
        const stream = result.stream.pipeThrough(
          new TransformStream<LanguageModelV3StreamPart, LanguageModelV3StreamPart>({
            flush() {
              onTrace({ preset: model.modelId, startedAt: new Date().toISOString(), usage: model.moaUsage() })
            },
          }),
        )
        return { ...result, stream }
      })
    },
  }
}
