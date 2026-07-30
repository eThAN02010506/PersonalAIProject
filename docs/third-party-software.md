# Third-Party Software

Qwopus-Agent itself currently grants no open-source license. Third-party
components remain governed by their own licenses; the absence of a Qwopus-Agent
license does not remove those obligations. This inventory is for engineering
review and is not legal advice.

## Integrated Components

| Component | Integration | License recorded by the project | Required attention |
| --- | --- | --- | --- |
| MinerU | `vendor/mineru` Git submodule | MinerU Open Source License, based on Apache-2.0 with additional terms | Third-party online services must prominently state that MinerU is used. Commercial thresholds and termination terms are defined in `vendor/mineru/LICENSE.md`. |
| MiniRAG | `vendor/minirag` and `minirag-hku` | MIT | Preserve the bundled copyright and license notice when redistributing its source. |
| smolagents | Python dependency | Apache-2.0 | Preserve required notices when redistributing the dependency or modified source. |
| sentence-transformers | Python dependency | Apache-2.0 | The library license does not replace the separate license of a downloaded embedding model. |
| assistant-ui and React packages | Frontend dependencies | Primarily MIT or ISC | Preserve notices required by the packages included in a distribution. |
| NetworkX, pandas, NumPy, pypdf and related Python libraries | Python dependencies | Primarily BSD, MIT, Apache-2.0, or project-specific permissive licenses | Re-run the dependency audit before distribution because unconstrained transitive versions can change. |

## Model Assets

Model weights are not part of the Qwopus-Agent source license. Gemma, Qwen,
Qwopus-derived models, embedding models, OCR models, and GGUF conversions may
each have separate terms. Record the exact model identifier, source, version,
and license before distributing weights or offering a service based on them.

## Release Check

Before making a build available outside the owner-controlled environment:

1. Confirm the pinned MinerU commit and read its bundled `LICENSE.md`.
2. Generate a fresh backend and frontend dependency inventory.
3. Verify the licenses of every downloaded generation, embedding, and OCR model.
4. Include required copyright, attribution, and notice text in the distribution.
5. Recheck whether the intended use crosses MinerU's online-service or commercial thresholds.
