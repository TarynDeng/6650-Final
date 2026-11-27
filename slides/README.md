# Slides

`6650_final_presentation.pptx` is the **draft** deck for the December 2 presentation (15 slides, about 13 minutes plus questions). It opens in PowerPoint, Keynote and Google Slides. Every slide has speaker notes with the numbers to say.

**Before presenting:** add the team names on slide 1, decide who presents which slides (see `docs/presentation.md`), and edit freely.

## Where the numbers come from

All charts are native PowerPoint charts, built from the experiment results, so nothing is typed in by hand:

```bash
python slides/export_deck_data.py      # reads results/ -> slides/deck_data.json
cd slides && npm install && npm run build   # deck_data.json -> 6650_final_presentation.pptx
```

Run these again after re-running the experiments. **Rebuilding overwrites manual edits to the .pptx**, so once the team starts editing the slides by hand, either stop rebuilding or make text changes in `build_deck.js` instead.
