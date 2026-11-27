// Results, conclusion and backup slides (called from build_deck.js).

module.exports = function addResults(ctx) {
  const { pres, D, L, slide, text, card, stat, bullets, barChart, pct, pp, pair, val, idx, N, AGENTS } = ctx;
  const { H, M, METHOD_COLORS, LOSS_COLORS, NEUTRAL, CHART_TEXT } = L;
  const C = pres.SchemeColor;
  const { oid } = ctx;
  const qbh = pair("Q-learning", "Buy-and-hold");
  const qd = pair("Q-learning", "DQN");
  const sd = pair("SARSA", "DQN");
  const we = pair("Weighted ens.", "Equal-weight ens.");
  const eq = pair("Equal-weight ens.", "Q-learning");
  const lossShare = (i) => (100 - D.loss_share.cat1[i]).toFixed(0);
  const pos = (m) => D.avg_position[idx[m]].toFixed(0);

  function rq1() {
    const s = slide("CONTENT", "Results");
    s.addText("RQ1: no RL method beat buy-and-hold in 2023–2025", { placeholder: "title" });
    barChart(s, D.mean_return, { name: "Mean test return (%)", chart: { x: M, y: 1.45, w: 7.6, h: 5.25,
      dataLabelFormatCode: '+0.0"%";-0.0"%"', valAxisLabelFormatCode: '0"%"',
      title: `Mean terminal return, test years (${N} stock-years, seeds averaged)` } });
    [[pp(qbh.diff), `Q-learning vs. buy-and-hold: Q-learning won ${qbh.wins} of ${qbh.n} stock-years`],
      [pp(qd.diff), `Q-learning vs. DQN: Q-learning won ${qd.wins} of ${qd.n}`],
      [pp(sd.diff), `SARSA vs. DQN: SARSA won ${sd.wins} of ${sd.n}`]]
      .forEach(([big, small], i) => stat(s, 8.55, 1.55 + i * 1.75, 4.15, 1.55, big, small));
    s.addNotes(`Buy-and-hold made ${pct(val("Buy-and-hold"))} on average. Q-learning made ${pct(val("Q-learning"))}; it held stock ${pos("Q-learning")}% of the time on average, versus ${pos("DQN")}% for DQN and ${pos("SARSA")}% for SARSA. DQN made ${pct(val("DQN"))}, SARSA ${pct(val("SARSA"))}. Q-learning beat DQN in ${qd.wins} of ${qd.n} stock-years, but not everywhere, so treat this as an indication, not proof.`);
  }

  function rq2() {
    const s = slide("CONTENT");
    s.addText("RQ2: no large losses, but the agents lost more often", { placeholder: "title" });
    s.addChart(pres.charts.BAR, [["Profit or break-even", "cat1"], ["Loss up to 20%", "cat2"],
      ["Loss 20–40%", "cat3"], ["Loss over 40%", "cat4"]].map(([name, k]) => ({ name, labels: D.methods,
      values: D.loss_share[k] })), {
      x: M, y: 1.45, w: 8.2, h: 5.25, barDir: "bar", barGrouping: "percentStacked", chartColors: LOSS_COLORS,
      catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "ctr", dataLabelFormatCode: '0"%";;;',
      showLegend: true, legendPos: "b", valAxisLabelFormatCode: "0%", showTitle: true,
      title: "Share of test episodes (stock-years × seeds) by terminal-loss category",
      ...CHART_TEXT, dataLabelColor: "1E2A26", objectName: oid("chart") });
    card(s, 9.15, 1.55, 3.55, 5.05);
    text(s, String(D.large_losses), { x: 9.45, y: 1.75, w: 3.0, h: 1.0, fontSize: 60, bold: true, color: C.accent1, fontFace: L.THEME.headFontFace });
    text(s, `episodes lost more than 20%. The worst loss was ${D.worst_loss.toFixed(1)}%.`, { x: 9.45, y: 2.8, w: 3.0, h: 0.9, fontSize: 16, valign: "top" });
    text(s, `Buy-and-hold lost in ${lossShare(0)}% of episodes; the agents in ${Math.min(...[1, 2, 3].map((i) => +lossShare(i)))}–${Math.max(...[1, 2, 3].map((i) => +lossShare(i)))}%. Only ${D.falling_stock_years.length} of the ${N} stock-years fell (${D.falling_stock_years.join(", ")}).`,
      { x: 9.45, y: 3.95, w: 3.0, h: 2.5, fontSize: 15, color: L.MUTED, valign: "top" });
    s.addNotes("Losses were small for everyone, but the RL agents ended more test years with a loss than buy-and-hold did, often by being out of the market in years the stock rose. With so few falling stock-years, RQ2 needs more bad markets to answer well.");
  }

  function rq3seeds() {
    const s = slide("CONTENT");
    s.addText("RQ3: DQN's results depend most on the random seed", { placeholder: "title" });
    const rows = [1, 2, 3, 4, 5];
    barChart(s, rows.map((i) => D.seed_std[i]), { name: "Spread across seeds", labels: rows.map((i) => D.methods[i]),
      colors: rows.map((i) => METHOD_COLORS[i]), chart: { x: M, y: 1.45, w: 7.6, h: 5.25, valAxisMinVal: 0,
        dataLabelFormatCode: '0.0" pp"', valAxisLabelFormatCode: "0",
        title: "Std. dev. of test return across 5 seeds (pp)" } });
    stat(s, 8.55, 1.55, 4.15, 2.3, `${D.seed_std[idx.DQN].toFixed(1)} pp`, "DQN: typical change in test return when only the random seed changes");
    stat(s, 8.55, 4.1, 4.15, 2.3, `${D.seed_std[idx.SARSA].toFixed(1)}–${D.seed_std[idx["Q-learning"]].toFixed(1)} pp`, "SARSA and Q-learning");
    s.addNotes("Same data, same settings, different seed: DQN's test return moves the most. The weighted ensemble inherits this, because it copies one agent.");
  }

  function rq3curves() {
    const s = slide("CONTENT");
    s.addText("RQ3: no steady improvement on validation", { placeholder: "title" });
    D.tickers.forEach((t, i) => {
      const cv = D.curves[t];
      s.addChart(pres.charts.LINE, AGENTS.map((a) => ({ name: a, labels: cv.steps.map((x) => `${x / 1000}k`), values: cv[a] })), {
        x: M + i * 4.07, y: 1.45, w: 3.95, h: 5.25, chartColors: METHOD_COLORS.slice(1, 4), lineSize: 2,
        lineDataSymbol: "none", showLegend: i === 0, legendPos: "b", valAxisLabelFormatCode: '0"%"',
        catAxisLabelFrequency: 5, showTitle: true, title: `${t}: 2022 return during training`,
        ...CHART_TEXT, objectName: oid("chart") });
    });
    s.addNotes(`Mean over ${D.seeds.length} seeds, x-axis in environment steps. On XOM both tabular agents get worse the longer they train, which is consistent with overfitting to 2015-2021; on AAPL SARSA improves. DQN is noisy throughout. One validation year per stock is too little to say more.`);
  }

  function rq4() {
    const s = slide("CONTENT");
    s.addText("RQ4: the weighted ensemble just copied the 2022 winner", { placeholder: "title" });
    const meanVal = (a) => D.tickers.reduce((acc, t) => acc + D.val_by_ticker[t][a], 0) / D.tickers.length;
    s.addChart(pres.charts.BAR, [
      { name: `2022 validation (mean of ${D.tickers.length} stocks)`, labels: AGENTS, values: AGENTS.map(meanVal) },
      { name: "2023–2025 test", labels: AGENTS, values: AGENTS.map((a) => D.test_by_agent[a]) }], {
      x: M, y: 1.45, w: 6.9, h: 5.25, barDir: "col", barGrouping: "clustered", chartColors: [NEUTRAL.light, NEUTRAL.dark],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '+0.0"%";-0.0"%"', valAxisLabelFormatCode: '0"%"',
      showLegend: true, legendPos: "b", showTitle: true, title: "Mean return of each agent: validation vs. test",
      ...CHART_TEXT, objectName: oid("chart") });
    card(s, 7.75, 1.55, 4.95, 2.4);
    text(s, "Agent the weighted ensemble followed", { x: 8.05, y: 1.7, w: 4.4, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
    text(s, AGENTS.map((a, i) => ({ text: `${a}: ${D.picks[a]} of ${D.n_runs} runs`, options: { breakLine: i < 2 } })),
      { x: 8.05, y: 2.2, w: 4.4, h: 1.6, fontSize: 18, valign: "top" });
    stat(s, 7.75, 4.15, 4.95, 2.45, `${pct(val("Weighted ens."))} vs. ${pct(val("Equal-weight ens."))}`,
      `Weighted vs. equal-weight. Equal weights won ${we.n - we.wins} of ${we.n} stock-years and finished within ${Math.abs(eq.diff).toFixed(1)} pp of Q-learning.`, 28);
    s.addNotes(`The weights come from one year, 2022, when the agents that did best were not the ones that did best in 2023-2025: Q-learning, the best test agent, was followed in only ${D.picks["Q-learning"]} of ${D.n_runs} runs. 2022 differed by stock (XOM rose about 80%), so this is a regime change, not a simple 'bear vs. bull' story. Tie-break caveat: when all three agents disagree, the equal-weight rule falls back to the lowest action number, which is usually a buy; with Hold instead its return would be ${pct(D.tiebreak.hold)} rather than ${pct(D.tiebreak.top_weight_then_hold)}.`);
  }

  function behavior() {
    const s = slide("CONTENT");
    s.addText("The agents held less stock and paid more fees", { placeholder: "title" });
    barChart(s, D.avg_position, { name: "Share of capital in stock", chart: { x: M, y: 1.45, w: 5.9, h: 5.25,
      valAxisMinVal: 0, valAxisMaxVal: 120, valAxisMajorUnit: 20, dataLabelFormatCode: '0"%"', valAxisLabelFormatCode: '0"%"',
      title: "Average share of capital held in stock" } });
    barChart(s, D.fees, { name: "Fees per year", chart: { x: 6.8, y: 1.45, w: 5.9, h: 5.25, valAxisMinVal: 0,
      dataLabelFormatCode: '"$"0', valAxisLabelFormatCode: '"$"0',
      title: `Fees per test year on $${D.config.cash.toLocaleString("en-US")} (average)` } });
    s.addNotes(`In rising markets, time out of the market and fees both cost return. DQN averaged ${D.trades[idx.DQN]} trades a year and about $${D.fees[idx.DQN]} in fees; buy-and-hold trades once.`);
  }

  function takeaways() {
    const s = slide("CLOSING_DARK", "Conclusion");
    s.addText("Takeaways and limitations", { placeholder: "title" });
    [`Q-learning beat DQN (${pct(val("Q-learning"))} vs. ${pct(val("DQN"))}); SARSA did not. Nothing beat buy-and-hold (${pct(val("Buy-and-hold"))}).`,
      "No large losses, but the agents ended more years with a small loss than buy-and-hold.",
      "DQN depends most on the random seed and trades the most.",
      `Weights from one validation year picked the wrong agent; equal weights did ${Math.abs(we.diff).toFixed(1)} pp better.`]
      .forEach((t, i) => {
        const y = 1.55 + i * 1.2;
        s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.62, h: 0.62, fill: { color: H.accent2 }, line: { color: H.accent2 }, objectName: oid("badge") });
        text(s, String(i + 1), { x: M, y, w: 0.62, h: 0.62, fontSize: 16, bold: true, align: "center", valign: "middle" });
        text(s, t, { x: M + 0.85, y: y - 0.05, w: 7.0, h: 0.85, fontSize: 18, color: C.background1, valign: "middle" });
      });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.85, y: 1.45, w: 3.85, h: 4.95, rectRadius: 0.12,
      fill: { color: "2B6350" }, line: { color: "2B6350" }, objectName: oid("card") });
    text(s, "Limitations", { x: 9.15, y: 1.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: C.background1 });
    text(s, bullets([`Only ${N} stock-years, and they move together`, "One validation year; no tuning",
      "DQN sees more information per step", "No spread or price impact"], 16),
    { x: 9.15, y: 2.25, w: 3.3, h: 3.9, color: C.background2, valign: "top" });
    s.addNotes("One line per research question. With this data a simple tabular agent was more reliable than DQN, and none of the agents added value over buying and holding. Nine correlated stock-years is a small test: these are indications, not proof.");
  }

  function backup() {
    const s = slide("CONTENT", "Backup");
    s.addText("Backup: settings and reproducibility", { placeholder: "title" });
    const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: H.accent1 } } });
    s.addTable([[hdr("Setting"), hdr("Value")], ["Stocks", D.tickers.join(", ")],
      ["Training / validation / test", "2015–2021 / 2022 / 2023, 2024, 2025"],
      ["Fee, execution", `${D.config.fee_pct}% per trade; fill at close t, value at close t+1`],
      ["Training budget", `${D.steps.toLocaleString("en-US")} steps per run, ${D.seeds.length} seeds, γ = ${D.config.gamma}`],
      ["Exploration", "ε from 1.0 to 0.05 over the first half of training"],
      ["Tabular", `${D.config.n_states} states, step size 0.05`],
      ["DQN", `${D.config.hidden.join("-")} ReLU, replay 50k, batch 64, lr 5e-4, soft target update 0.005`],
      ["Ensemble", `softmax, T = ${D.config.temperature}, on 2022 return; ties: top weight, Hold, lowest action`]],
    { x: M, y: 1.45, w: 7.9, colW: [2.6, 5.3], fontSize: 13, color: H.dk1, rowH: 0.5, valign: "middle",
      border: { type: "solid", pt: 0.75, color: "DDE3DF" }, objectName: oid("table") });
    card(s, 8.85, 1.45, 3.85, 4.5);
    text(s, "Reproduce", { x: 9.15, y: 1.65, w: 3.3, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
    text(s, bullets(["github.com/TarynDeng/6650-Final", `Results from commit ${D.commit}`,
      "3 commands in the README", `Prices committed with checksums; ${D.n_tests} automated tests`], 14),
    { x: 9.15, y: 2.25, w: 3.3, h: 3.5, valign: "top" });
    s.addNotes("Backup for questions about settings. Every setting is in src/config.py, and the README explains how to reproduce each number in this deck.");
  }

  [rq1, rq2, rq3seeds, rq3curves, rq4, behavior, takeaways, backup].forEach((f) => f());
};
