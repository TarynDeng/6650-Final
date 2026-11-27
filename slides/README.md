# Slides

`6650_final_presentation.pptx` is the **draft** deck for the December 2 presentation (15 slides, about 13 minutes plus questions). It opens in PowerPoint, Keynote and Google Slides, and every slide has speaker notes. The slide-by-slide plan, with an Owner column for splitting the talk, is in [`docs/presentation.md`](../docs/presentation.md).

**Before presenting:** add the team names on slide 1 and decide who presents which slides.

## How it is built

Every number and chart comes from the committed results; nothing is typed in by hand:

```bash
python slides/export_deck_data.py            # results/ -> slides/deck_data.json
cd slides && npm install && npm run build    # deck_data.json -> 6650_final_presentation.pptx
```

`build_deck.js` holds the introduction and method slides, `deck_results.js` the results and closing slides, and `deck_layout.js` the theme, layouts and shared helpers.

**Rebuilding overwrites manual edits to the .pptx.** Once the team starts editing the slides by hand, either stop rebuilding or make the change in the scripts instead.
