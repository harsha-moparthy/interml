# Teaching guide for ML lessons

Status: Implemented design direction.

## Purpose

Help the learner explain, predict, and implement a computation.
Use guided discovery first. Then allow free exploration.
Choose the teaching method for the concept and the question contract.
An animation must expose a mechanism or a change that the learner needs to understand.

## Lesson sequence

1. Present a concrete puzzle with a small input.
2. State the learning goal and required prior knowledge.
3. Guide the learner through one complete example.
4. Pause before a meaningful step. Ask for a prediction.
5. Reveal the result and explain its cause.
6. Let the learner change an input and inspect the effect.
7. Introduce a useful failure case or remove a component.
8. Connect the result to the equations and solution code.
9. Give a new example with less guidance.
10. Ask the learner to explain the result and attempt the implementation.

Reveal detail gradually. Keep advanced derivations available without crowding the first view.
For abstract ML mechanisms, show a concept story before raw arrays. Start with named
objects, roles, ownership, and information flow. Then provide an explicit control to
inspect vector bars, geometry, and only then exact numerical values when required. Never imply that an illustrative semantic label
is a literal learned embedding dimension.
Allow the learner to skip familiar steps and return to them.
Keep prediction answers hidden until the learner requests them.
Explain incorrect predictions with a specific example. Avoid generic success or failure messages.

## Teaching methods

| Method | Learner action | Authoring requirement |
| --- | --- | --- |
| Build an operation | Assemble axes, graph nodes, masks, or update terms | Explain why each part is necessary; offer keyboard controls |
| Predict and reveal | Predict a shape, value, or next state | Give a checked answer and explain the reasoning |
| Compare two cases | Change one condition while keeping other inputs fixed | Identify the changed condition and its measurable effect |
| Break and repair | Introduce a specific error and correct it | Connect the observed failure to a violated rule |
| Trace one object | Follow a cell, vector, token, or gradient | Preserve its identity across all relevant operations |
| Switch representations | Move from concept story to geometry, values, equations, and code | Keep the selected object and computation state synchronized |
| Remove a component | Disable a mask, scaling term, cache, or normalization step | Explain the resulting behavior and the limits of the comparison |
| Solve a new case | Apply the idea to an unseen example | Change meaningful inputs; include an explained answer |

Select methods that fit the lesson. Do not require every method in every lesson.
Every deep lesson needs a guided example, a prediction, an inspectable computation, and a new case.
Include a useful failure case or a controlled comparison.

## Choose the representation

| Concept | Useful representation | Detail to expose |
| --- | --- | --- |
| Shapes and indexing | Labeled arrays, 3D axes, selectable 2D slices | Input coordinates for a selected output cell |
| Geometry | Vectors, contours, rotations, and linked equations | Numerical values and preserved quantities |
| Information flow | Graphs and matrix traces | Where values originate and how they combine |
| Time and state | Timelines with discrete steps | State before and after each operation |
| Optimization | Objective plots and update traces | Contributions of individual update terms |
| Probability | Distributions and small enumerated examples | Normalization, likelihood, and uncertainty |
| Systems behavior | Event traces and resource tables | Ordering, work, memory, and failure boundaries |

Use 3D when it explains axes or geometry. Provide a readable 2D alternative.
Use a static diagram when motion adds no useful information.
Use toy values that make arithmetic readable. Identify all simplifications.
Define the limits of any analogy before connecting it to the actual computation.

## Interaction rules

Provide play, pause, forward, backward, and reset for ordered demonstrations.
Provide a labeled timeline when the example has several stages.
Advance one meaningful operation at a time.
Explain what changed, why it changed, and what remained fixed.
Show values and shapes on selection. Keep essential information available without hover.
Allow the learner to select one object and trace its complete path.
Keep colors, labels, variable names, and symbols consistent across views.
Keep editable inputs small enough for direct inspection.
Show invalid-input explanations beside the relevant control.
Do not start important animations automatically.
Provide keyboard controls, reduced motion, and static descriptions.

Each demonstration uses one computation state for values, equations, plots, and code highlights.
Use reproducible examples. Preserve the seed when comparing randomized cases.
Stepping backward must restore the earlier state accurately.
Reset must restore the documented starting inputs and state.
Check displayed values against a reference computation.

## Help at the current step

Provide a control named “Help me understand this step”.
Offer authored explanations that apply to the current operation:

- Show a smaller numerical example.
- Explain the prerequisite.
- Show the geometry or a static diagram.
- Expand the equation with substituted values.
- Show a common mistake and its consequence.
- Connect this step to the solution code.

Show only relevant options. Preserve the learner's current state when help opens.
Link each help item to its step and prerequisite article.
These explanations are authored content. They do not require an AI tutor.

## Broadcasting: construct and trace

Use question 030 with a prerequisite article on broadcasting.
Let the learner align dimensions and insert a singleton axis.
Show `(2, 1, 3) + (1, 4, 1) -> (2, 4, 3)` in 3D and in selectable slices.
Select output coordinate `(i, j, k)` to show `A[i, 0, k] + B[0, j, 0]`.
Display the selected input values and the output calculation.
Explain conceptual expansion separately from input memory allocation.
Connect array views and shared storage to question 027.
Ask the learner to construct all pairwise differences by inserting axes.
Include incompatible shapes. Identify the exact dimensions that conflict.

## Attention: trace and change

Use question 316 for multi-head attention and question 328 for cross-attention.
Select one query token and follow its full computation.
Show projections, scores, softmax weights, weighted values, head concatenation, and output projection.
Use different query and source lengths for cross-attention.
Show the source of each Q, K, and V vector explicitly.
Offer a “Why this output?” trace from one output component to its contributors.
In a controlled example, edit a projected key while holding other projected vectors fixed.
Then edit a projected value. Explain the different effects on weights and output.
Distinguish these controlled edits from changing an input embedding, which can affect several projections.
Include a mask error that the learner can repair.
Use fixed illustrative values. Do not claim they represent learned language relationships.

## KV cache: compare two timelines

Use question 391 with a short text example.
Run cached and uncached computations with identical weights, positions, and token prefixes.
Show prompt processing and each later generation step.
Highlight reused keys and values, appended entries, and the current query.
Ask which entries must change before revealing the next state.
Check output agreement within a stated numerical tolerance.
Let the learner introduce an incorrect position offset and locate the first divergent result.
Display operation counts and cache memory separately.
Label measured runtime only when the program actually measures it.

## RoPE: discover the relationship

Use question 331. Match its vector pairing and frequency conventions.
Show vector pairs, position controls, rotation equations, and substituted numbers together.
Move both positions by the same amount while preserving their separation and frequencies.
Then change their separation. Inspect the query-key dot product in both cases.
Connect the observed result to the relative-position derivation.
Check norm preservation and numerical results.
End with a new vector pair and ask the learner to predict the rotation.

## Backpropagation: account for contributions

Use questions 108 and 113 with a small computation graph.
Trace local derivatives and incoming gradients one operation at a time.
Show separate contributions when a value reaches the output along multiple paths.
Ask the learner to add those contributions before advancing.
Connect broadcast reduction to the original input shape.
Compare the result with a numerical gradient where applicable.
Include an example that loses a contribution by overwriting an accumulated gradient.

## Evidence of completion

Before marking a lesson complete, record these checks:

- The guided example agrees with the question contract.
- Displayed values agree with an independent calculation or reference operation.
- Selection links agree across diagrams, equations, values, and code.
- Forward, backward, reset, and input edits produce consistent states.
- The prediction and new case have checked, explained answers.
- The failure case or comparison demonstrates the intended cause.
- Step help and prerequisite links resolve to complete content.
- Keyboard use, reduced motion, and static alternatives work.

Technical checks do not prove that a learner understands the concept.
Keep learner feedback separate from implementation verification.
Record an observed misunderstanding and revise the explanation when feedback is available.
