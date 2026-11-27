// Builds slides/6650_final_presentation.pptx from slides/deck_data.json.
//
//   python slides/export_deck_data.py        # results/ -> deck_data.json
//   cd slides && npm install && npm run build
//
// Every number on the slides comes from deck_data.json. Rebuilding overwrites
// manual edits to the .pptx (see slides/README.md).

const fs = require("fs");
const path = require("path");
const L = require("./deck_layout");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "6650_final_presentation.pptx");
const pres = L.newPresentation();
const { C, oid, text, card, stat, bullets } = L.helpers(pres);
const { H, M, MUTED, METHOD_COLORS, CHART_TEXT } = L;

const pct = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}%`;
const pp = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)} pp`;
const pair = (a, b) => D.paired.find((p) => p.a === a && p.b === b);
const idx = Object.fromEntries(D.methods.map((m, i) => [m, i]));
const val = (m) => D.mean_return[idx[m]];
const N = D.n_stock_years;
const AGENTS = ["Q-learning", "SARSA", "DQN"];
let section = "";
const slide = (master, sec) => {
  if (sec && sec !== section) { pres.addSection({ title: sec }); section = sec; }
  return pres.addSlide({ masterName: master, sectionTitle: section });
};
const barChart = (s, values, opts) => s.addChart(pres.charts.BAR, [{ name: opts.name, labels: opts.labels || D.methods,
  values }], { barDir: "bar", chartColors: opts.colors || METHOD_COLORS, invertedColors: opts.colors || METHOD_COLORS,
  catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", showLegend: false, showTitle: true,
  ...CHART_TEXT, objectName: oid("chart"), ...opts.chart });

function titleSlide() {
  const s = slide("TITLE_DARK", "Introduction");
  s.addText("Shallow vs. Deep RL in Stock Trading", { placeholder: "title" });
  s.addText([
    { text: "A comparative study of Q-learning, SARSA and DQN, with a performance-weighted ensemble",
      options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 10 } },
    { text: "DSAN 6650 Reinforcement Learning · Final Project · December 2, 2026",
      options: { fontSize: 16, color: "C9D6CF", breakLine: true } },
    { text: "Team: add names", options: { fontSize: 16, color: "C9D6CF" } },
  ], { placeholder: "body" });
  s.addNotes("Introduce the team and the question: does a deep RL agent trade better than tabular RL when everything else is held equal? Replace 'Team: add names' with the four presenters.");
}

function motivationSlide() {
  const s = slide("CONTENT");
  s.addText("Neural network or lookup table: which trades better?", { placeholder: "title" });
  text(s, bullets([
    "The course covers tabular RL (Q-learning, SARSA) and deep RL (DQN). We compare them on one task, with the data, actions, budget and evaluation held equal.",
    "Stock trading is a realistic test: noisy rewards, a changing market, and cheap to simulate on historical data.",
    "The goal is to study the algorithms, not to build a trading product. All trades are simulated.",
  ], 18), { x: M, y: 1.7, w: 6.6, h: 4.6, valign: "top" });
  [["3", "RL algorithms"], ["2", "ensembles of them"], ["9", "actions: hold + 8 trade sizes"], ["1", "buy-and-hold benchmark"]]
    .forEach(([big, small], i) => {
      const x = 7.7 + (i % 2) * 2.6, y = 1.7 + Math.floor(i / 2) * 2.35;
      card(s, x, y, 2.35, 2.05);
      text(s, big, { x: x + 0.2, y: y + 0.2, w: 1.95, h: 1.05, fontSize: 54, bold: true, color: C.accent1,
        fontFace: L.THEME.headFontFace, align: "center", valign: "middle" });
      text(s, small, { x: x + 0.2, y: y + 1.3, w: 1.95, h: 0.6, fontSize: 15, align: "center" });
    });
  s.addNotes("Deep RL gets the headlines, but our homework used both kinds of methods. We ask whether a neural network's extra capacity pays off on noisy market data. Trading is only the testbed.");
}

function questionsSlide() {
  const s = slide("CONTENT");
  s.addText("Four research questions", { placeholder: "title" });
  [["RQ1 · Performance", "How does shallow (tabular) RL compare with deep RL in trading return?"],
    ["RQ2 · Terminal losses", "How often, and how badly, does each method end a test year with a loss?"],
    ["RQ3 · Learning and stability", "How fast do they learn, and how much do results change with the random seed?"],
    ["RQ4 · Ensemble", "Does a vote weighted by validation performance beat the single agents and an equal-weight vote?"]]
    .forEach(([head, q], i) => {
      const x = M + (i % 2) * 6.15, y = 1.55 + Math.floor(i / 2) * 2.65;
      card(s, x, y, 5.95, 2.4, { badge: String(i + 1) });
      text(s, head, { x: x + 1.15, y: y + 0.3, w: 4.6, h: 0.62, fontSize: 20, bold: true, color: C.accent1, valign: "middle" });
      text(s, q, { x: x + 1.15, y: y + 1.0, w: 4.55, h: 1.25, fontSize: 18, valign: "top" });
    });
  s.addNotes("The four questions from our proposal. RQ2 is about the loss at the end of a test year, not the maximum drawdown during it.");
}

function dataSlide() {
  const s = slide("CONTENT", "Method");
  s.addText("Data: three stocks, eleven years, split by date", { placeholder: "title" });
  text(s, `Daily split- and dividend-adjusted prices from Yahoo Finance for ${D.tickers.join(", ")} (technology, banking, energy).`,
    { x: M, y: 1.45, w: 12.1, h: 0.45 });
  const yw = 12.1 / 11;
  [[0, 7, "Training 2015–2021", H.accent1], [7, 1, "Validation 2022", H.accent2], [8, 1, "Test 2023", H.accent4],
    [9, 1, "Test 2024", H.accent4], [10, 1, "Test 2025", H.accent4]].forEach(([start, len, label, col]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: M + start * yw + 0.03, y: 2.35, w: len * yw - 0.06, h: 0.95,
      fill: { color: col }, line: { color: col }, objectName: oid("timeline") });
    text(s, label, { x: M + start * yw, y: 2.35, w: len * yw, h: 0.95, fontSize: 14, bold: true,
      color: C.background1, align: "center", valign: "middle" });
  });
  [[0, 7, "Train the agents"], [7, 1, "Ensemble weights"], [8, 3, "Never seen in training"]].forEach(([a, n, t]) =>
    text(s, t, { x: M + a * yw - 0.2, y: 3.4, w: n * yw + 0.4, h: 0.35, fontSize: 13, color: MUTED, align: "center" }));
  [[`$${D.config.cash.toLocaleString("en-US")}`, "starting cash, one stock per episode"],
    [`${D.config.fee_pct}%`, "fee on every trade"], ["No", "leverage or short selling"],
    ["Close t", "order fills; valued at close t+1 (no look-ahead)"]]
    .forEach(([big, small], i) => stat(s, M + i * 3.05, 4.15, 2.85, 2.45, big, small));
  s.addNotes("Prices are committed to the repository with checksums, so everyone uses identical data. Training uses random one-year windows inside 2015-2021; 2022 only sets the ensemble weights; the three test years are never used in training. Orders fill at today's close and the portfolio is valued at tomorrow's close.");
}

function mdpSlide() {
  const s = slide("CONTENT");
  s.addText("What the agent sees, does, and is rewarded for", { placeholder: "title" });
  card(s, M, 1.5, 3.7, 5.1);
  text(s, "State", { x: M + 0.3, y: 1.7, w: 3.1, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  text(s, bullets(["5-day return", "20-day return", "20-day volatility", "Share of the portfolio held in stock"]),
    { x: M + 0.3, y: 2.3, w: 3.15, h: 2.4, valign: "top" });
  text(s, `Tabular agents: ${D.config.n_states} discrete states. DQN: the 4 numbers, standardized.`,
    { x: M + 0.3, y: 4.85, w: 3.15, h: 1.5, fontSize: 14, color: MUTED, valign: "top" });
  text(s, "Actions: direction and size together", { x: 4.65, y: 1.55, w: 4.6, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
  const action = (x, y, w, col, label) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.62, rectRadius: 0.08, fill: { color: col },
      line: { color: col }, objectName: oid("action") });
    text(s, label, { x: x + 0.15, y, w: w - 0.2, h: 0.62, fontSize: 14, bold: true, color: C.background1, valign: "middle" });
  };
  action(4.65, 2.15, 4.45, H.accent6, "0  Hold");
  [25, 50, 75, 100].forEach((p, i) => {
    action(4.65, 2.95 + i * 0.85, 2.15, H.accent1, `${i + 1}  Buy ${p}% of cash`);
    action(6.95, 2.95 + i * 0.85, 2.15, "7A2E21", `${i + 5}  Sell ${p}% of shares`);
  });
  card(s, 9.55, 1.5, 3.15, 5.1);
  text(s, "Reward", { x: 9.85, y: 1.7, w: 2.6, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  text(s, "rₜ = ln(Vₜ₊₁ / Vₜ)", { x: 9.85, y: 2.4, w: 2.65, h: 0.7, fontSize: 24, fontFace: "Cambria" });
  text(s, `Daily log return of the portfolio value V (cash + shares), fees included.\n\nUndiscounted, a year's rewards add up to ln(V_T / V_0). The agents use γ = ${D.config.gamma}, which looks about 100 days ahead.`,
    { x: 9.85, y: 3.25, w: 2.6, h: 3.1, fontSize: 14, valign: "top" });
  s.addNotes("The state mixes three market signals with how invested we are. The 9 actions pick direction and size at once. The reward is the daily log return; its undiscounted sum is the log of final wealth, but with gamma 0.99 the agents effectively plan about 100 trading days ahead.");
}

function algorithmsSlide() {
  const s = slide("CONTENT");
  s.addText("Three agents, one shared training loop", { placeholder: "title" });
  [["Q", "Q-learning", "Tabular, off-policy", "Learns from the best next action."],
    ["S", "SARSA", "Tabular, on-policy", "Learns from the next action it actually takes."],
    ["D", "DQN", "Neural network, off-policy", `${D.config.hidden.join("-")} network, replay buffer, target network. Based on our HW-2 code.`]]
    .forEach(([b, name, kind, desc], i) => {
      const x = M + i * 4.1;
      card(s, x, 1.5, 3.85, 3.0, { badge: b, badgeColor: METHOD_COLORS[i + 1] });
      text(s, name, { x: x + 1.15, y: 1.8, w: 2.5, h: 0.62, fontSize: 22, bold: true, color: C.accent1, valign: "middle" });
      text(s, kind, { x: x + 0.3, y: 2.6, w: 3.3, h: 0.4, fontSize: 16, bold: true });
      text(s, desc, { x: x + 0.3, y: 3.05, w: 3.3, h: 1.3, fontSize: 16, valign: "top" });
    });
  text(s, "Same for every agent", { x: M, y: 4.85, w: 12.1, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
  [[`${D.steps / 1000}k`, "trading days of experience"], [`${D.config.episode_len}`, "days per training episode"],
    [`γ ${D.config.gamma}`, "discount factor"], [`${D.seeds.length}`, "random seeds per stock"]].forEach(([big, small], i) => {
    text(s, big, { x: M + i * 3.05, y: 5.4, w: 2.85, h: 0.65, fontSize: 30, bold: true, color: C.accent1, fontFace: L.THEME.headFontFace });
    text(s, small, { x: M + i * 3.05, y: 6.05, w: 2.85, h: 0.45, fontSize: 14, color: MUTED });
  });
  s.addNotes("Q-learning and SARSA differ only in the update target. DQN replaces the table with a neural network. All three share the training loop and are evaluated with their final model; nothing was tuned. Caveat: DQN sees continuous features and learns from a 64-sample batch every step, so 'deep vs. shallow' also means 'more information per step'.");
}

function ensembleSlide() {
  const s = slide("CONTENT");
  s.addText("Ensembles: let the three agents vote every day", { placeholder: "title" });
  card(s, M, 1.5, 6.2, 5.1);
  text(s, "Weighted ensemble", { x: M + 0.3, y: 1.7, w: 5.6, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  text(s, "wᵢ = exp(Rᵢ / T) / Σⱼ exp(Rⱼ / T)", { x: M + 0.3, y: 2.35, w: 5.6, h: 0.6, fontSize: 22, fontFace: "Cambria" });
  text(s, `Rᵢ = agent i's return in 2022, temperature T = ${D.config.temperature}`, { x: M + 0.3, y: 2.95, w: 5.6, h: 0.4, fontSize: 14, color: MUTED });
  text(s, "aₜ = argmaxₐ Σᵢ wᵢ · 1[πᵢ(sₜ) = a]", { x: M + 0.3, y: 3.65, w: 5.6, h: 0.6, fontSize: 22, fontFace: "Cambria" });
  text(s, "The action with the most total weight wins.", { x: M + 0.3, y: 4.25, w: 5.6, h: 0.4, fontSize: 14, color: MUTED });
  text(s, "Equal-weight ensemble: the same vote with every weight 1/3. This is the control.",
    { x: M + 0.3, y: 5.0, w: 5.6, h: 1.3, fontSize: 16, valign: "top" });
  text(s, "Rules fixed before seeing results", { x: 7.2, y: 1.6, w: 5.5, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  text(s, bullets([
    "Weights use 2022 only; the test years are never used.",
    "All three agents decide from the ensemble's own cash and position.",
    "Ties: follow the highest-weight agent, then Hold, then the lowest action number.",
    `In practice, with T = ${D.config.temperature}, one agent got over half the weight in ${D.follows_top} of ${D.n_runs} runs, so the weighted "vote" simply follows that agent.`,
  ]), { x: 7.2, y: 2.25, w: 5.5, h: 4.3, valign: "top" });
  s.addNotes("A small temperature makes the weights decisive; in our runs the top weight was always above 0.5, so the weighted ensemble is effectively 'pick the agent that did best in 2022'. The equal-weight vote tells us whether that selection helps.");
}

[titleSlide, motivationSlide, questionsSlide, dataSlide, mdpSlide, algorithmsSlide, ensembleSlide].forEach((f) => f());
require("./deck_results")({ pres, D, L, oid, slide, text, card, stat, bullets, barChart, pct, pp, pair, val, idx, N, AGENTS });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await L.applyTheme(OUT);
  console.log(`Wrote ${OUT}`);
})();
