// Shared computation state for demonstrations. Pure functions, no React.
// State holds only the inputs, the step index, and the selection. Frames are derived from the inputs,
// so stepping backward restores the earlier state exactly.

export function initialState(spec, defaults, inputs) {
  const sanitized = spec.sanitize(inputs ?? defaults);
  return { inputs: sanitized, step: 0, selection: spec.defaultSelection(sanitized), playing: false, helpOpen: false };
}

export function framesOf(spec, state) {
  return spec.compute(state.inputs, state.selection).frames;
}

export function makeReducer(spec, defaults) {
  const last = (inputs) => spec.compute(inputs).frames.length - 1;
  return function reducer(state, action) {
    switch (action.type) {
      case "next": {
        const step = Math.min(state.step + 1, last(state.inputs));
        return { ...state, step, playing: state.playing && step < last(state.inputs) };
      }
      case "prev": return { ...state, step: Math.max(state.step - 1, 0), playing: false };
      case "goto": return { ...state, step: Math.min(Math.max(action.step, 0), last(state.inputs)), playing: false };
      case "reset": return initialState(spec, defaults);
      case "setInput": {
        const inputs = spec.sanitize({ ...state.inputs, [action.name]: action.value });
        return {
          ...state, inputs, playing: false,
          step: Math.min(state.step, last(inputs)),
          selection: spec.normalizeSelection(inputs, state.selection),
        };
      }
      case "select": return { ...state, selection: spec.normalizeSelection(state.inputs, action.selection) };
      case "play": return state.step >= last(state.inputs) ? state : { ...state, playing: true };
      case "pause": return { ...state, playing: false };
      case "toggleHelp": return { ...state, helpOpen: !state.helpOpen };
      default: return state;
    }
  };
}
