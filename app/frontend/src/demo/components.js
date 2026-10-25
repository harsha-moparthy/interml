// UI side of the registry: spec (pure) plus React views.
import { REGISTRY_SPECS } from "./registry.js";
import { BroadcastView, BroadcastInputs, BroadcastCompareView } from "./BroadcastView.jsx";
import { ConceptLabView, ConceptInputs } from "./ConceptLabView.jsx";
import { makeFramesSpec } from "./frames.js";

export const COMPONENTS = {
  "broadcast-trace": { spec: REGISTRY_SPECS["broadcast-trace"], View: BroadcastView, Inputs: BroadcastInputs, CompareView: BroadcastCompareView },
  "concept-lab": { makeSpec: makeFramesSpec, View: ConceptLabView, Inputs: ConceptInputs },
};
