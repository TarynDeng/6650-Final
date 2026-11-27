// Theme, slide layouts, and drawing helpers shared by build_deck.js.

const fs = require("fs");
const pptxgen = require("pptxgenjs");

const THEME = {
  name: "6650 Final",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2A26", lt1: "FFFFFF", dk2: "1F4E3D", lt2: "EEF3EF",
    accent1: "1F6E4E", accent2: "E0A43A", accent3: "6FA287",
    accent4: "2F6F8F", accent5: "C9772B", accent6: "8A8F8C",
    hlink: "2F6F8F", folHlink: "5F6B66",
  },
};
const H = THEME.colors;
const MUTED = "5F6B66";
// One color per method on every chart: buy-and-hold, Q-learning, SARSA, DQN, weighted, equal.
const METHOD_COLORS = [H.accent6, H.accent1, H.accent3, H.accent4, H.accent2, H.accent5];
// Colors for series that are not methods (never reuse a method color).
const LOSS_COLORS = ["C9D6CF", "B5533C", "7A2E21", "3D150F"];
const NEUTRAL = { light: "B8C4BE", dark: "3E4A45" };
const W = 13.333;
const M = 0.6;

function newPresentation() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.title = "Shallow vs. Deep RL in Stock Trading";
  pres.subject = "DSAN 6650 Final Project";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  defineLayouts(pres);
  return pres;
}

function defineLayouts(pres) {
  const C = pres.SchemeColor;
  const title = (color) => ({ placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: 12.1,
    h: 0.95, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color, valign: "middle", align: "left" },
  text: "" } });
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: H.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 2.1, w: 12.1, h: 1.6,
        fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom" },
      text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: 12.1, h: 2.4,
        fontFace: THEME.bodyFontFace, fontSize: 20, color: C.background2, valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: H.lt1 },
    margin: [0.5, M, 0.6, M],
    objects: [title(C.accent1),
      { text: { text: "DSAN 6650 · Shallow vs. Deep RL in Stock Trading",
        options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } }],
    slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: H.dk2 },
    objects: [title(C.background1)],
    slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: "C9D6CF", align: "right" },
  });
}

// Drawing helpers bound to one presentation.
function helpers(pres) {
  const C = pres.SchemeColor;
  let objId = 0;
  const oid = (p) => `${p}-${++objId}`;
  const text = (slide, t, o) => slide.addText(t, { isTextBox: true, margin: 0, fontSize: 16,
    color: C.text1, objectName: oid("text"), ...o });
  // A tinted card with an optional numbered circle (the deck's visual motif).
  const card = (slide, x, y, w, h, opts = {}) => {
    const fill = opts.fill || H.lt2;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
      line: { color: fill }, objectName: oid("card") });
    if (opts.badge) {
      const col = opts.badgeColor || H.accent1;
      slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fill: { color: col },
        line: { color: col }, objectName: oid("badge") });
      text(slide, opts.badge, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fontSize: 16, bold: true,
        color: C.background1, align: "center", valign: "middle" });
    }
  };
  // A card with a big number (or short phrase) and a caption under it.
  const stat = (slide, x, y, w, h, big, small, size = 30) => {
    card(slide, x, y, w, h);
    text(slide, big, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.75, fontSize: size, bold: true,
      color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
    text(slide, small, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: h - 1.0, fontSize: 14, valign: "top" });
  };
  const bullets = (items, size = 16) => items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, fontSize: size } }));
  return { C, oid, text, card, stat, bullets };
}

const CHART_TEXT = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: H.dk1,
  titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: H.dk1,
  legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: H.dk1,
  valGridLine: { color: "DDE3DF", size: 0.75 }, catGridLine: { style: "none" } };

// pptxgenjs writes Office's default palette into the theme; write ours, drop
// folder entries some cloud services reject, and fix one chart-axis element
// PowerPoint would otherwise offer to "repair".
async function applyTheme(file) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file), { createFolders: false });
  for (const name of Object.keys(zip.files)) if (zip.files[name].dir) delete zip.files[name];
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5",
    "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${THEME.name}">` +
    slots.map((k) => `<a:${k}><a:srgbClr val="${H[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const themePath = "ppt/theme/theme1.xml";
  const xml = await zip.file(themePath).async("string");
  if (!/<a:clrScheme\b/.test(xml)) throw new Error("theme has no color scheme to replace");
  zip.file(themePath, xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, scheme), { createFolders: false });
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const c = await zip.file(name).async("string");
    zip.file(name, c.replaceAll('<c:crossesAt val="autoZero"/>', '<c:crosses val="autoZero"/>'),
      { createFolders: false });
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { THEME, H, MUTED, METHOD_COLORS, LOSS_COLORS, NEUTRAL, W, M, CHART_TEXT,
  newPresentation, helpers, applyTheme };
