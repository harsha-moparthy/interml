# Broadcasting article redesign

## Delivered surface

Route: `http://127.0.0.1:5173/#/article/lesson-q030-pairwise-differences`.
The question index links to the replacement article.
The original solution is loaded from `/api/exercises/30/solution`.
The new article uses a dedicated React presentation. The generic article renderer and question bank remain available.
The pilot prose is in `app/frontend/src/lessons/BroadcastArticle.jsx`.
Moving this presentation into reusable database-backed article blocks is follow-up work after the pilot review.

## Replacement and recovery

There is no Git repository in the workspace. A historical rollback was not possible.
The prior frontend source and question 030 article were saved in `backups/before-broadcast-redesign.tar.gz`.
The old question 030 presentation is bypassed at its existing route.
No question records or original solutions were deleted.

## Design reference

Built-in Image Gen generated `broadcast-concept.png`.
The brief requested a complete primary broadcasting article screen with downstream article styling.
It specified a white background, navy editorial typography, teal A, amber B, and violet output.
It specified the headline “One rule. Every pair.” and the subtitle shown in the implementation.
It required real React controls and mathematically computed diagrams.
The generated image is a visual reference, not a source of numerical truth.

Tokens: white #ffffff; navy #14223e; teal #08776e; amber #9a5d00; violet #7139ba.
Type: Georgia headings and Inter/Segoe UI body. Buttons use explicit 14px type.
Spacing uses a 1320px desktop container, 34px inner gutters, and open article sections.

## Visual comparison

The concept and settled browser screenshots were inspected with the image viewer.
Playwright Chromium was used because no built-in browser tool was available.
Desktop checks used 1440 × 1100 and the concept's native 1435 × 1096 size.
Mobile checks used 390 × 844.

| Comparison | Result |
| --- | --- |
| White surface and navy editorial heading | Retained |
| Teal input, amber input, violet result | Retained consistently across diagrams and computation trace |
| Header, headline, subtitle, and question navigation | Matched the intended wording and hierarchy |
| Canvas, playback row, and explanation band | Retained; controls execute real state changes |
| Diagram labels and selected arithmetic | Corrected numerical and axis inaccuracies from the generated concept |
| Tensor legibility | Increased row spacing after screenshot review to remove overlapping values |
| Narrow layout | Text stacks; the wide diagram scrolls within its own region without page overflow |

The design direction was faithfully checked against the concept, with intentional functional deviations.
The implementation uses exploded isometric tiles with an adjustable depth angle, not a fully orbitable WebGL scene.
This keeps every number readable and makes keyboard selection possible.
The canvas is taller than the concept to expose all 24 output values.
Help, shape repair, pair selection, and prediction controls were added to meet the user's teaching requirements.
The above-the-fold copy retains the headline, subtitle, brand, and navigation.
The explanation changes with the current animation step. Values are computed instead of copied from the mockup.

## Functional evidence

- Production build passed. The existing large-chunk warning remains.
- All 30 frontend tests passed.
- The new shape checker agrees with 304 stored NumPy reference cases.
- All 24 animated output cells agree with the NumPy reference values.
- Browser checks passed for cell tracing, edited input values, reset, play/pause, and step navigation.
- Browser checks passed for projected/flat views, shape failure/repair, pair subtraction, and prediction feedback.
- The original PDF solution loads through the existing API and can be hidden again.
- Mobile overflow and reduced-motion behavior passed. No page errors were reported.

Screenshots are visual evidence, not proof that a learner understands the topic.
The next step is review of this one article before expanding the design to other lessons.
