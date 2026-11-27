// Builds slides/6650_final_presentation.pptx from slides/deck_data.json.
//
//   python slides/export_deck_data.py   # numbers from results/
//   cd slides && npm install && npm run build
//
// Every number on the slides comes from deck_data.json, so re-running the
// experiments and these two commands updates the whole deck.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "6650_final_presentation.pptx");

// ---------------------------------------------------------------- theme
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
// One color per method, used on every chart (buy-and-hold, Q, SARSA, DQN, weighted, equal).
const METHOD_COLORS = [H.accent6, H.accent1, H.accent3, H.accent4, H.accent2, H.accent5];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Shallow vs. Deep RL in Stock Trading";
pres.subject = "DSAN 6650 Final Project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;

// ---------------------------------------------------------------- layouts
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
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: 12.1, h: 0.95,
      fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.accent1, valign: "middle", align: "left" },
      text: "" } },
    { text: { text: "DSAN 6650 · Shallow vs. Deep RL in Stock Trading",
      options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
});

pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: 12.1, h: 0.95,
      fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left" },
      text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: "C9D6CF", align: "right" },
});

// ---------------------------------------------------------------- helpers
let objId = 0;
const oid = (p) => `${p}-${++objId}`;

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, objectName: oid("text"), ...o });
}

// A tinted card with an optional numbered circle (the deck's visual motif).
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: opts.fill || H.lt2 }, line: { color: opts.fill || H.lt2 },
    objectName: oid("card"),
  });
  if (opts.badge) {
    slide.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62,
      fill: { color: opts.badgeColor || H.accent1 }, line: { color: opts.badgeColor || H.accent1 },
      objectName: oid("badge") });
    text(slide, opts.badge, { x: x + 0.3, y: y + 0.3, w: 0.62, h: 0.62, fontSize: 16, bold: true,
      color: C.background1, align: "center", valign: "middle" });
  }
}

function bullets(items, size = 16) {
  return items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8, fontSize: size },
  }));
}

const CHART_TEXT = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: THEME.colors.dk1,
  titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: THEME.colors.dk1,
  legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: THEME.colors.dk1 };
const QUIET_GRID = { valGridLine: { color: "DDE3DF", size: 0.75 }, catGridLine: { style: "none" } };

const pct = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const pp = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)} pp`;
const find = (a, b) => D.paired.find((p) => p.a === a && p.b === b);
const M_IDX = Object.fromEntries(D.methods.map((m, i) => [m, i]));
const nWin = D.tickers.length * 3;

// ================================================================ slides
// 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Shallow vs. Deep RL in Stock Trading", { placeholder: "title" });
s.addText([
  { text: "A comparative study of Q-learning, SARSA and DQN, with a performance-weighted ensemble",
    options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 10 } },
  { text: "DSAN 6650 Reinforcement Learning · Final Project · December 2, 2026",
    options: { fontSize: 16, color: "C9D6CF", breakLine: true } },
  { text: "Team: add names", options: { fontSize: 16, color: "C9D6CF" } },
], { placeholder: "body" });
s.addNotes("Introduce the team and the one-line question: does a deep RL agent trade better than simple tabular RL when everything else is held equal? Replace 'Team: add names' with the four presenters.");

// 2. Motivation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Neural network or lookup table: which trades better?", { placeholder: "title" });
text(s, bullets([
  "The course covers tabular RL (Q-learning, SARSA) and deep RL (DQN). We compare them on the same task, with everything else held equal.",
  "Stock trading is a realistic test: noisy rewards, a changing market, and cheap to simulate on historical data.",
  "The goal is to study the algorithms, not to build a trading product. All trades are simulated.",
], 18), { x: M, y: 1.7, w: 6.6, h: 4.6, valign: "top", fontSize: 18 });
const stats = [["3", "RL algorithms"], ["2", "ensembles of them"], ["9", "trade actions"], ["1", "buy-and-hold benchmark"]];
stats.forEach(([big, small], i) => {
  const x = 7.7 + (i % 2) * 2.6, y = 1.7 + Math.floor(i / 2) * 2.35;
  card(s, x, y, 2.35, 2.05);
  text(s, big, { x: x + 0.2, y: y + 0.2, w: 1.95, h: 1.05, fontSize: 54, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace, align: "center", valign: "middle" });
  text(s, small, { x: x + 0.2, y: y + 1.3, w: 1.95, h: 0.55, fontSize: 15, color: C.text1, align: "center" });
});
s.addNotes("Motivation. Deep RL gets the headlines, but our homework used both kinds of methods. We ask whether the extra capacity of a neural network pays off on noisy market data. Trading is only the testbed.");

// 3. Research questions
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Four research questions", { placeholder: "title" });
const rqs = [
  ["RQ1", "Performance", "How does shallow (tabular) RL compare with deep RL in trading return?"],
  ["RQ2", "Terminal losses", "How often, and how badly, does each method end a test year with a loss?"],
  ["RQ3", "Learning and stability", "How fast do they learn, and how much do results change with the random seed?"],
  ["RQ4", "Ensemble", "Does a vote weighted by validation performance beat the single agents and an equal-weight vote?"],
];
rqs.forEach(([id, head, q], i) => {
  const x = M + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.65;
  card(s, x, y, 5.95, 2.4, { badge: String(i + 1) });
  text(s, `${id} · ${head}`, { x: x + 1.15, y: y + 0.3, w: 4.6, h: 0.62, fontSize: 20, bold: true,
    color: C.accent1, valign: "middle" });
  text(s, q, { x: x + 1.15, y: y + 1.0, w: 4.55, h: 1.25, fontSize: 18, valign: "top" });
});
s.addNotes("The four questions from our proposal. RQ2 is about the loss at the end of a test year, not the maximum drawdown during it.");

// 4. Data and environment
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Data: three stocks, eleven years, split by date", { placeholder: "title" });
text(s, `Daily split- and dividend-adjusted prices from Yahoo Finance for ${D.tickers.join(", ")} (technology, banking, energy).`,
  { x: M, y: 1.45, w: 12.1, h: 0.45 });
// Timeline 2015-2025, proportional to years.
const tl = { x: M, y: 2.35, w: 12.1, h: 0.95 }, yw = tl.w / 11;
[[0, 7, "Training 2015–2021", H.accent1], [7, 1, "Validation 2022", H.accent2],
 [8, 1, "Test 2023", H.accent4], [9, 1, "Test 2024", H.accent4], [10, 1, "Test 2025", H.accent4]]
  .forEach(([start, len, label, col]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: tl.x + start * yw + 0.03, y: tl.y, w: len * yw - 0.06, h: tl.h,
      fill: { color: col }, line: { color: col }, objectName: oid("timeline") });
    text(s, label, { x: tl.x + start * yw, y: tl.y, w: len * yw, h: tl.h, fontSize: 14, bold: true,
      color: C.background1, align: "center", valign: "middle" });
  });
text(s, "Train the agents", { x: tl.x, y: 3.4, w: 7 * yw, h: 0.35, fontSize: 13, color: MUTED, align: "center" });
text(s, "Ensemble weights", { x: tl.x + 7 * yw - 0.2, y: 3.4, w: yw + 0.4, h: 0.35, fontSize: 13, color: MUTED, align: "center" });
text(s, "Never seen in training", { x: tl.x + 8 * yw, y: 3.4, w: 3 * yw, h: 0.35, fontSize: 13, color: MUTED, align: "center" });
const rules = [["$10,000", "starting cash, one stock per episode"], ["0.1%", "fee on every trade"],
  ["No", "leverage or short selling"], ["Close t", "order fills; valued at close t+1 (no look-ahead)"]];
rules.forEach(([big, small], i) => {
  const x = M + i * 3.05;
  card(s, x, 4.15, 2.85, 2.45);
  text(s, big, { x: x + 0.25, y: 4.35, w: 2.35, h: 0.85, fontSize: 30, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace, valign: "middle" });
  text(s, small, { x: x + 0.25, y: 5.25, w: 2.35, h: 1.15, fontSize: 15, valign: "top" });
});
s.addNotes("Data come from Yahoo Finance and are committed to the repository with checksums, so every teammate uses identical prices. Training uses random one-year windows inside 2015-2021. 2022 is only used to set the ensemble weights. The three test years are never touched during training. Orders fill at today's close, the portfolio is valued at tomorrow's close, so no future information leaks in.");

// 5. State, actions, reward
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("What the agent sees, does, and is rewarded for", { placeholder: "title" });
card(s, M, 1.5, 3.7, 5.1);
text(s, "State", { x: M + 0.3, y: 1.7, w: 3.1, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
text(s, bullets(["5-day return", "20-day return", "20-day volatility", "Share of the portfolio held in stock"], 16),
  { x: M + 0.3, y: 2.3, w: 3.15, h: 2.4, valign: "top" });
text(s, "Tabular agents: 135 discrete states. DQN: the 4 numbers, standardized.", { x: M + 0.3, y: 4.85, w: 3.15,
  h: 1.5, fontSize: 14, color: MUTED, valign: "top" });

text(s, "Actions: direction and size together", { x: 4.65, y: 1.55, w: 4.6, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.65, y: 2.15, w: 4.45, h: 0.6, rectRadius: 0.08,
  fill: { color: H.accent6 }, line: { color: H.accent6 }, objectName: oid("action") });
text(s, "0  Hold", { x: 4.8, y: 2.15, w: 4.2, h: 0.6, fontSize: 15, bold: true, color: C.background1, valign: "middle" });
[25, 50, 75, 100].forEach((p, i) => {
  const y = 2.95 + i * 0.85;
  [["Buy", H.accent1, i + 1, "of cash"], ["Sell", H.accent5, i + 5, "of shares"]].forEach(([side, col, n, of], j) => {
    const x = 4.65 + j * 2.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.15, h: 0.65, rectRadius: 0.08,
      fill: { color: col }, line: { color: col }, objectName: oid("action") });
    text(s, `${n}  ${side} ${p}% ${of}`, { x: x + 0.15, y, w: 1.95, h: 0.65, fontSize: 14, bold: true,
      color: C.background1, valign: "middle" });
  });
});

card(s, 9.55, 1.5, 3.15, 5.1);
text(s, "Reward", { x: 9.85, y: 1.7, w: 2.6, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
text(s, "rₜ = ln(Vₜ₊₁ / Vₜ)", { x: 9.85, y: 2.4, w: 2.65, h: 0.7, fontSize: 24, fontFace: "Cambria", color: C.text1 });
text(s, "Daily log return of the portfolio value V (cash + shares), fees included.\n\nOver a year the rewards add up to ln(V_T / V_0), so more reward means more final wealth. No extra risk penalty.",
  { x: 9.85, y: 3.25, w: 2.6, h: 3.1, fontSize: 14, valign: "top" });
s.addNotes("The state mixes three market signals with how invested we currently are. The 9 actions let the agent pick timing, direction and size at once. The reward is the daily log return, which sums exactly to the log of final wealth over an episode, so the agent is optimizing the thing we measure.");

// 6. Algorithms
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Three agents, one shared training loop", { placeholder: "title" });
const algos = [
  ["Q", "Q-learning", "Tabular, off-policy", "Learns from the best next action. 135-state table."],
  ["S", "SARSA", "Tabular, on-policy", "Learns from the next action it actually takes. Same 135 states."],
  ["D", "DQN", "Neural network, off-policy", "64-64 network, replay buffer, target network. Based on our HW-2 code."],
];
algos.forEach(([b, name, kind, desc], i) => {
  const x = M + i * 4.1;
  card(s, x, 1.5, 3.85, 3.0, { badge: b, badgeColor: METHOD_COLORS[i + 1] });
  text(s, name, { x: x + 1.15, y: 1.8, w: 2.5, h: 0.62, fontSize: 22, bold: true, color: C.accent1, valign: "middle" });
  text(s, kind, { x: x + 0.3, y: 2.6, w: 3.3, h: 0.4, fontSize: 15, bold: true });
  text(s, desc, { x: x + 0.3, y: 3.05, w: 3.3, h: 1.3, fontSize: 15, valign: "top" });
});
text(s, "Same for every agent", { x: M, y: 4.85, w: 12.1, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
const same = [[`${D.steps / 1000}k`, "trading days of experience"], ["252", "days per training episode"],
  ["γ 0.99", "discount factor"], [`${D.seeds.length}`, "random seeds per stock"]];
same.forEach(([big, small], i) => {
  const x = M + i * 3.05;
  text(s, big, { x, y: 5.4, w: 2.85, h: 0.65, fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  text(s, small, { x, y: 6.05, w: 2.85, h: 0.45, fontSize: 14, color: MUTED });
});
s.addNotes("Q-learning and SARSA differ only in the update target. DQN replaces the table with a neural network. To keep the comparison fair, all three use one training loop: same step budget, same exploration schedule, same discount, same seeds, and each is evaluated with its final model.");

// 7. Weighted ensemble
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Ensembles: let the three agents vote every day", { placeholder: "title" });
card(s, M, 1.5, 6.2, 5.1);
text(s, "Weighted ensemble", { x: M + 0.3, y: 1.7, w: 5.6, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
text(s, "wᵢ = exp(R̄ᵢ / τ) / Σⱼ exp(R̄ⱼ / τ)", { x: M + 0.3, y: 2.35, w: 5.6, h: 0.6, fontSize: 22, fontFace: "Cambria" });
text(s, "R̄ᵢ = agent i's return in the 2022 validation year, τ = 0.05", { x: M + 0.3, y: 2.95, w: 5.6, h: 0.4, fontSize: 14, color: MUTED });
text(s, "aₜ = argmaxₐ Σᵢ wᵢ · 1[πᵢ(sₜ) = a]", { x: M + 0.3, y: 3.65, w: 5.6, h: 0.6, fontSize: 22, fontFace: "Cambria" });
text(s, "The action with the most total weight wins.", { x: M + 0.3, y: 4.25, w: 5.6, h: 0.4, fontSize: 14, color: MUTED });
text(s, "Equal-weight ensemble: the same vote with every weight 1/3. This is the control.",
  { x: M + 0.3, y: 5.0, w: 5.6, h: 1.3, fontSize: 16, valign: "top" });
text(s, "Rules fixed before seeing results", { x: 7.2, y: 1.6, w: 5.5, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
text(s, bullets([
  "Weights use 2022 only; the test years are never used.",
  "All three agents decide from the ensemble's own cash and position, not from their own backtests.",
  "Ties: follow the highest-weight agent, then Hold.",
]), { x: 7.2, y: 2.25, w: 5.5, h: 3.8, valign: "top" });
s.addNotes("The weight of each agent is a softmax of its 2022 return. A small temperature makes the weights decisive. The ensemble trades one shared account, and every member re-decides on that account every day. The equal-weight vote is the control that tells us whether the weighting helps.");

// 8. Setup
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("How we compare the six methods", { placeholder: "title" });
const setup = [["6", "methods: 3 agents, 2 ensembles, buy-and-hold"], [`${D.tickers.length} × 3`, "stocks × test years"],
  [`${nWin}`, "stock-years: the unit of comparison"], [`${D.seeds.length}`, "seeds, averaged within each stock-year"]];
setup.forEach(([big, small], i) => {
  const x = M + i * 3.05;
  card(s, x, 1.5, 2.85, 2.3);
  text(s, big, { x: x + 0.25, y: 1.7, w: 2.35, h: 0.9, fontSize: 36, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace, valign: "middle" });
  text(s, small, { x: x + 0.25, y: 2.65, w: 2.35, h: 1.0, fontSize: 15, valign: "top" });
});
text(s, "Metrics", { x: M, y: 4.15, w: 6, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
text(s, bullets([
  "Terminal return (V_T − V_0) / V_0 at the end of each test year",
  "Terminal-loss category: profit, loss up to 20%, 20–40%, over 40% (not max drawdown)",
  "Spread across seeds; validation return during training; trades and fees",
], 15), { x: M, y: 4.65, w: 12.1, h: 2.0, valign: "top" });
s.addNotes("Seeds repeat the same market, so we average them first and treat each stock-year as one sample. That gives 9 samples, and they are not independent: stocks in the same year move together. Keep that in mind for every result that follows.");

// 9. RQ1
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("RQ1: no RL method beat buy-and-hold in 2023–2025", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Mean test return (%)", labels: D.methods, values: D.mean_return }], {
  x: M, y: 1.45, w: 7.6, h: 5.25, barDir: "bar", chartColors: METHOD_COLORS, invertedColors: METHOD_COLORS,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '+0.0"%";-0.0"%"',
  valAxisLabelFormatCode: '0"%"', catAxisOrientation: "maxMin", valAxisMinVal: 0, showLegend: false,
  showTitle: true, title: `Mean terminal return, test years (${nWin} stock-years, seeds averaged)`,
  ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
const qbh = find("Q-learning", "Buy-and-hold"), qd = find("Q-learning", "DQN"), sd = find("SARSA", "DQN");
[[pp(qbh.diff), `Q-learning vs. buy-and-hold; Q-learning won ${qbh.wins} of ${qbh.n} stock-years`],
 [pp(qd.diff), `Q-learning vs. DQN; Q-learning won ${qd.wins} of ${qd.n}`],
 [pp(sd.diff), `SARSA vs. DQN: about even (${sd.wins} of ${sd.n})`]]
  .forEach(([big, small], i) => {
    const y = 1.55 + i * 1.75;
    card(s, 8.55, y, 4.15, 1.55);
    text(s, big, { x: 8.8, y: y + 0.15, w: 3.7, h: 0.7, fontSize: 30, bold: true, color: C.accent1,
      fontFace: THEME.headFontFace, valign: "middle" });
    text(s, small, { x: 8.8, y: y + 0.85, w: 3.7, h: 0.6, fontSize: 14, valign: "top" });
  });
s.addNotes(`Buy-and-hold made ${pct(D.mean_return[0])} on average. The best agent, Q-learning, made ${pct(D.mean_return[1])}; it stays mostly invested. DQN made ${pct(D.mean_return[3])} and SARSA ${pct(D.mean_return[2])}. Tabular Q-learning beat DQN in ${qd.wins} of ${qd.n} stock-years. In a rising market, any time out of the market and any fee costs return.`);

// 10. RQ2
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("RQ2: losses were small, and similar across methods", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Profit or break-even", labels: D.methods, values: D.loss_share.cat1 },
  { name: "Loss up to 20%", labels: D.methods, values: D.loss_share.cat2 },
  { name: "Loss 20–40%", labels: D.methods, values: D.loss_share.cat3 },
  { name: "Loss over 40%", labels: D.methods, values: D.loss_share.cat4 },
], {
  x: M, y: 1.45, w: 8.2, h: 5.25, barDir: "bar", barGrouping: "percentStacked",
  chartColors: [H.accent3, H.accent2, H.accent5, "8E3B2F"], catAxisOrientation: "maxMin",
  showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: '0"%";;;', dataLabelColor: "FFFFFF",
  showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0%",
  showTitle: true, title: "Share of test episodes in each terminal-loss category",
  ...CHART_TEXT, dataLabelColor: "FFFFFF", ...QUIET_GRID, objectName: oid("chart"),
});
card(s, 9.15, 1.55, 3.55, 5.05);
text(s, "0", { x: 9.45, y: 1.75, w: 3.0, h: 1.0, fontSize: 60, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
text(s, "episodes lost more than 20% of their capital.", { x: 9.45, y: 2.8, w: 3.0, h: 0.9, fontSize: 16, valign: "top" });
text(s, `Only 1 of the ${nWin} stock-years fell (XOM 2023), so the test period says little about bad markets.`,
  { x: 9.45, y: 3.95, w: 3.0, h: 2.4, fontSize: 15, color: MUTED, valign: "top" });
s.addNotes(`Between ${Math.min(...D.loss_share.cat2)}% and ${Math.max(...D.loss_share.cat2)}% of episodes ended with a small loss, and none lost more than 20%. Categories 3 and 4 never occurred. This is mostly a property of the test years, which were strong for these stocks, so RQ2 needs more falling markets to answer well.`);

// 11. RQ3
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("RQ3: DQN varies most by seed; longer training can overfit", { placeholder: "title" });
const cv = D.curves[D.tickers[D.tickers.length - 1]];
s.addChart(pres.charts.LINE, ["Q-learning", "SARSA", "DQN"].map((a) => ({
  name: a, labels: cv.steps.map((x) => `${x / 1000}k`), values: cv[a] })), {
  x: M, y: 1.45, w: 6.9, h: 5.25, chartColors: [H.accent1, H.accent3, H.accent4], lineSize: 2.5,
  lineDataSymbol: "none", showLegend: true, legendPos: "b", valAxisLabelFormatCode: '0"%"',
  showTitle: true, title: `${D.tickers[D.tickers.length - 1]}: 2022 validation return during training (mean of ${D.seeds.length} seeds)`,
  catAxisLabelFrequency: 4, ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
const stdIdx = [1, 2, 3, 4, 5];
s.addChart(pres.charts.BAR, [{ name: "Spread across seeds", labels: stdIdx.map((i) => D.methods[i]),
  values: stdIdx.map((i) => D.seed_std[i]) }], {
  x: 7.75, y: 1.45, w: 4.95, h: 5.25, barDir: "bar", chartColors: stdIdx.map((i) => METHOD_COLORS[i]),
  invertedColors: stdIdx.map((i) => METHOD_COLORS[i]), catAxisOrientation: "maxMin", valAxisMinVal: 0,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0" pp"', valAxisLabelFormatCode: "0",
  showLegend: false, showTitle: true, title: "Spread of test return across seeds (pp)",
  ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
s.addNotes(`Left: on XOM, both tabular agents get worse on 2022 the longer they train, a sign of overfitting to 2015-2021. DQN's curve is noisy throughout. Right: changing only the random seed moves DQN's test return by ${D.seed_std[3]} points on average, versus ${D.seed_std[2]} for SARSA and ${D.seed_std[1]} for Q-learning.`);

// 12. RQ4
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("RQ4: validation weights picked the wrong agents", { placeholder: "title" });
const ag = ["Q-learning", "SARSA", "DQN"];
s.addChart(pres.charts.BAR, [
  { name: "2022 validation (falling market)", labels: ag, values: ag.map((a) => D.val_return[a]) },
  { name: "2023–2025 test (rising market)", labels: ag, values: ag.map((a) => D.mean_return[M_IDX[a]]) },
], {
  x: M, y: 1.45, w: 6.9, h: 5.25, barDir: "col", barGrouping: "clustered", chartColors: [H.accent5, H.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '+0.0"%";-0.0"%"', valAxisLabelFormatCode: '0"%"',
  showLegend: true, legendPos: "b", showTitle: true, title: "Mean return of each agent: validation year vs. test years",
  ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
const we = find("Weighted ens.", "Equal-weight ens.");
card(s, 7.75, 1.55, 4.95, 2.4);
text(s, "Average weight", { x: 8.05, y: 1.7, w: 4.4, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
text(s, [
  { text: `Q-learning ${D.avg_weights.qlearning.toFixed(2)}`, options: { breakLine: true } },
  { text: `SARSA ${D.avg_weights.sarsa.toFixed(2)}`, options: { breakLine: true } },
  { text: `DQN ${D.avg_weights.dqn.toFixed(2)}` },
], { x: 8.05, y: 2.2, w: 4.4, h: 1.6, fontSize: 18, valign: "top" });
card(s, 7.75, 4.15, 4.95, 2.45);
text(s, `${pct(D.mean_return[4])} vs. ${pct(D.mean_return[5])}`, { x: 8.05, y: 4.3, w: 4.4, h: 0.7, fontSize: 28, bold: true,
  color: C.accent1, fontFace: THEME.headFontFace, valign: "middle" });
text(s, `Weighted vs. equal-weight ensemble. Equal weights won ${we.n - we.wins} of ${we.n} stock-years and matched Q-learning.`,
  { x: 8.05, y: 5.05, w: 4.4, h: 1.4, fontSize: 15, valign: "top" });
s.addNotes(`2022 was a falling year for two of our stocks, so the agents that held less stock looked best on validation and got most of the weight. In 2023-2025 the market rose and the agent that stays invested, Q-learning, did best, but it had the smallest weight. Weighting by one past year assumes the next years look like it; here the regime flipped. The equal-weight vote avoided that and finished at ${pct(D.mean_return[5])}.`);

// 13. Trading behavior
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("DQN trades the most and pays the most in fees", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Trades per year", labels: D.methods, values: D.trades }], {
  x: M, y: 1.45, w: 5.9, h: 5.25, barDir: "bar", chartColors: METHOD_COLORS, invertedColors: METHOD_COLORS,
  catAxisOrientation: "maxMin", valAxisMinVal: 0, showValue: true, dataLabelPosition: "outEnd", showLegend: false,
  showTitle: true, title: "Trades per test year (average)", ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
s.addChart(pres.charts.BAR, [{ name: "Fees per year", labels: D.methods, values: D.fees }], {
  x: 6.8, y: 1.45, w: 5.9, h: 5.25, barDir: "bar", chartColors: METHOD_COLORS, invertedColors: METHOD_COLORS,
  catAxisOrientation: "maxMin", valAxisMinVal: 0, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: '"$"0', valAxisLabelFormatCode: '"$"0', showLegend: false,
  showTitle: true, title: "Fees per test year on $10,000 (average)", ...CHART_TEXT, ...QUIET_GRID, objectName: oid("chart"),
});
s.addNotes(`DQN averaged ${D.trades[3]} trades a year and about $${D.fees[3]} in fees, roughly ${(D.fees[3] / 100).toFixed(1)}% of capital, while buy-and-hold trades once. Frequent trading is part of why the agents trail buy-and-hold: fees compound and the agent is often partly out of a rising market.`);

// 14. Takeaways
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
s.addText("Takeaways and limitations", { placeholder: "title" });
const take = [
  ["1", `Simple beats deep here: Q-learning ${pct(D.mean_return[1])} vs. DQN ${pct(D.mean_return[3])}, but nobody beat buy-and-hold (${pct(D.mean_return[0])}).`],
  ["2", "Losses were small for every method; the test years had almost no falling markets."],
  ["3", "DQN is the most sensitive to the random seed and trades the most."],
  ["4", "Weights from one validation year failed when the market regime changed; equal weights were safer."],
];
take.forEach(([n, t], i) => {
  const y = 1.55 + i * 1.2;
  s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.62, h: 0.62, fill: { color: H.accent2 }, line: { color: H.accent2 },
    objectName: oid("badge") });
  text(s, n, { x: M, y, w: 0.62, h: 0.62, fontSize: 16, bold: true, color: C.text1, align: "center", valign: "middle" });
  text(s, t, { x: M + 0.85, y: y - 0.05, w: 7.0, h: 0.85, fontSize: 18, color: C.background1, valign: "middle" });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.85, y: 1.45, w: 3.85, h: 4.95, rectRadius: 0.12,
  fill: { color: "2B6350" }, line: { color: "2B6350" }, objectName: oid("card") });
text(s, "Limitations", { x: 9.15, y: 1.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: C.background1 });
text(s, bullets([`Only ${nWin} stock-years, and they move together`, "Mostly rising test market",
  "One validation year", "No spread or price impact"], 17), { x: 9.15, y: 2.25, w: 3.3, h: 3.9, color: C.background2, valign: "top" });
s.addNotes("One line per research question. The honest summary: with this data, a simple tabular agent was more reliable than DQN, and none of the agents added value over buying and holding in a bull market. The limitations matter: 9 correlated samples is a small test, so these are indications, not proof.");

// 15. Backup
pres.addSection({ title: "Backup" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: settings and reproducibility", { placeholder: "title" });
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: H.accent1 } } });
s.addTable([
  [hdr("Setting"), hdr("Value")],
  ["Stocks", D.tickers.join(", ")],
  ["Training / validation / test", "2015–2021 / 2022 / 2023, 2024, 2025"],
  ["Fee, execution", "0.1% per trade; fill at close t, value at close t+1"],
  ["Training budget", `${D.steps.toLocaleString("en-US")} steps per run, ${D.seeds.length} seeds, γ = 0.99`],
  ["Exploration", "ε from 1.0 to 0.05 over the first half of training"],
  ["Tabular", "3 bins per feature × 5 position levels = 135 states, step size 0.05"],
  ["DQN", "64-64 ReLU, replay 50k, batch 64, lr 5e-4, soft target update 0.005"],
  ["Ensemble", "softmax τ = 0.05 on 2022 return; ties: top weight, then Hold"],
], { x: M, y: 1.45, w: 7.9, colW: [2.6, 5.3], fontSize: 13, color: THEME.colors.dk1,
  border: { type: "solid", pt: 0.75, color: "DDE3DF" }, rowH: 0.5, valign: "middle", objectName: oid("table") });
card(s, 8.85, 1.45, 3.85, 4.5);
text(s, "Reproduce", { x: 9.15, y: 1.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
text(s, bullets(["github.com/TarynDeng/6650-Final", `Results from commit ${D.commit}`, "3 commands, about 15 minutes on a laptop",
  "Prices committed with checksums; 39 automated tests"], 14), { x: 9.15, y: 2.25, w: 3.3, h: 3.5, valign: "top" });
s.addNotes("Backup slide for questions about settings. Everything is in src/config.py in the repository, and the README explains how to reproduce every number in this deck.");

// ---------------------------------------------------------------- write
// pptxgenjs always writes Office's default color palette into the theme, so the
// scheme colors used above would show up as Office blue. Write our palette into
// ppt/theme/theme1.xml, and fix one chart-axis element PowerPoint would "repair".
async function applyTheme(file, theme) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file), { createFolders: false });
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5",
    "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` +
    slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  // pptxgenjs also writes explicit folder entries, which some cloud services choke on.
  for (const name of Object.keys(zip.files)) if (zip.files[name].dir) delete zip.files[name];
  const themePath = "ppt/theme/theme1.xml";
  const xml = await zip.file(themePath).async("string");
  if (!/<a:clrScheme\b/.test(xml)) throw new Error("theme has no color scheme to replace");
  zip.file(themePath, xml.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, scheme), { createFolders: false });
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const c = await zip.file(name).async("string");
    zip.file(name, c.replaceAll('<c:crossesAt val="autoZero"/>', '<c:crosses val="autoZero"/>'), { createFolders: false });
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log(`Wrote ${OUT}`);
})();
