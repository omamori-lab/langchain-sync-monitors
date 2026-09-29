# Initial implementation

The first design of the library: control monitors as LangChain middleware,
with Defer to Resample from Ctrl-Z and an Auto Mode steering monitor as the
first protocols.

- `plan.html`: open it in a browser. It has the architecture diagrams,
  readable Python pseudocode for every component, the pre-build checks and
  their results, and every decision taken with the owner.
- `research/ctrl-z-defer-to-resample-spec.md`: the Defer to Resample protocol
  as the Ctrl-Z paper specifies it, with page references.
- `research/guard-model-scoring.md`: how a guard model's label becomes a
  suspicion score, from the literature.
- `research/decision-model-question-format.md`: which question a decision
  model should answer, from the literature.

Built in pull requests #28 to #34; the issues are grouped under the Phase 0 and
Phase 1 milestones.
