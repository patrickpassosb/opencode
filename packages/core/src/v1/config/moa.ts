export * as ConfigMoAV1 from "./moa"

import { Schema } from "effect"
import { PositiveInt } from "../../schema"

export interface Advisor extends Schema.Schema.Type<typeof Advisor> {}
export const Advisor = Schema.Struct({
  provider: Schema.String.annotate({ description: "Provider ID for the advisor model" }),
  model: Schema.String.annotate({ description: "Model ID for the advisor" }),
  maxTokens: Schema.optional(PositiveInt).annotate({
    description: "Caps the advisor analysis output in tokens. Defaults to the preset reference_max_tokens.",
  }),
}).annotate({ identifier: "MoA.Advisor" })

export interface Aggregator extends Schema.Schema.Type<typeof Aggregator> {}
export const Aggregator = Schema.Struct({
  provider: Schema.String.annotate({ description: "Provider ID for the aggregator model" }),
  model: Schema.String.annotate({ description: "Model ID for the aggregator" }),
  maxTokens: Schema.optional(PositiveInt).annotate({
    description: "Caps the final synthesized answer in tokens. Defaults to the preset max_tokens.",
  }),
}).annotate({ identifier: "MoA.Aggregator" })

export const FanoutMode = Schema.Literals(["user_turn", "stream"]).annotate({
  identifier: "MoA.FanoutMode",
  description: "user_turn fans out once per user turn; stream fans out per streamed segment.",
})

export interface Preset extends Schema.Schema.Type<typeof Preset> {}
export const Preset = Schema.Struct({
  advisors: Schema.Array(Advisor).annotate({
    description: "Advisor models consulted in parallel on each fan-out.",
  }),
  aggregator: Aggregator.annotate({
    description: "Model that synthesizes advisor analyses into the final answer.",
  }),
  maxTokens: Schema.optional(PositiveInt).annotate({
    description: "Maximum tokens for the aggregator output (default: 4096).",
  }),
  referenceMaxTokens: Schema.optional(PositiveInt).annotate({
    description: "Maximum tokens per advisor analysis (default: 600).",
  }),
  fanout: Schema.optional(FanoutMode).annotate({
    description: "When to fan out advisor calls (default: user_turn).",
  }),
  temperature: Schema.optional(Schema.Finite).annotate({
    description: "Sampling temperature applied to all preset calls (default: provider default).",
  }),
}).annotate({ identifier: "MoA.Preset" })

export const Info = Schema.Struct({
  default_preset: Schema.optional(Schema.String).annotate({
    description: "Preset used when the picker selection has no explicit preset.",
  }),
  presets: Schema.Record(Schema.String, Preset).annotate({
    description: "Named MoA presets, exposed in the model picker as moa/<name>.",
  }),
  save_traces: Schema.optional(Schema.Boolean).annotate({
    description: "Persist advisor/aggregator traces for debugging (default: false).",
  }),
}).annotate({ identifier: "MoA.Config" })

export type Info = Schema.Schema.Type<typeof Info>
