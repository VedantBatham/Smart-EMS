/**
 * Plotly.js Charts Module for Smart Energy Management System
 * Renders high-performance, dark-themed, responsive interactive visualizations.
 */

const darkLayoutDefaults = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  height: 380,
  autosize: true,
  font: { family: "'Outfit', 'Inter', sans-serif", color: "#94a3b8", size: 12 },
  margin: { l: 55, r: 40, t: 35, b: 45 },
  legend: { orientation: "h", y: 1.15, x: 0.5, xanchor: "center", font: { color: "#f8fafc", size: 11 } },
  xaxis: {
    gridcolor: "rgba(255, 255, 255, 0.06)",
    zerolinecolor: "rgba(255, 255, 255, 0.12)",
    tickfont: { color: "#64748b" },
  },
  yaxis: {
    gridcolor: "rgba(255, 255, 255, 0.06)",
    zerolinecolor: "rgba(255, 255, 255, 0.12)",
    tickfont: { color: "#64748b" },
  },
};

const Charts = {
  plot(containerId, traces, layout, config = { responsive: true, displayModeBar: false }) {
    if (typeof Plotly === "undefined") {
      console.warn("Plotly library is not loaded yet.");
      return;
    }
    const el = document.getElementById(containerId);
    if (!el) {
      console.warn(`Chart container #${containerId} not found in DOM.`);
      return;
    }
    try {
      Plotly.purge(el); // Purge any existing axis state or corrupted cache
      Plotly.newPlot(el, traces, layout, config);
    } catch (e) {
      console.error(`Error plotting #${containerId}:`, e);
    }
  },

  renderSolarVsLoad(containerId, timestamps, solar, load) {
    const traces = [
      {
        x: timestamps,
        y: solar,
        name: "Solar PV Generation (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: "#f59e0b", width: 2.2 },
        fill: "tozeroy",
        fillcolor: "rgba(245, 158, 11, 0.08)",
      },
      {
        x: timestamps,
        y: load,
        name: "Household Demand (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: "#06b6d4", width: 2.2 },
      },
    ];

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: { ...darkLayoutDefaults.yaxis, title: "Power (kW)" },
    };

    this.plot(containerId, traces, layout);
  },

  renderForecast(containerId, timestamps, actual, predicted, label, actualColor, predColor) {
    const traces = [
      {
        x: timestamps,
        y: actual,
        name: `Actual ${label} (kW)`,
        type: "scatter",
        mode: "lines",
        line: { color: actualColor, width: 2 },
      },
      {
        x: timestamps,
        y: predicted,
        name: `XGBoost Predicted ${label} (kW)`,
        type: "scatter",
        mode: "lines",
        line: { color: predColor, width: 2, dash: "dot" },
      },
    ];

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: { ...darkLayoutDefaults.yaxis, title: "Power (kW)" },
    };

    this.plot(containerId, traces, layout);
  },

  getRlLabel(viewMode) {
    if (viewMode === "hybrid_dqn_ppo" || viewMode === "hybrid") return "Hybrid DQN-PPO";
    if (viewMode === "best") return "Hybrid DQN-PPO (Best RL)";
    return "Forecast-Assisted PPO";
  },

  renderBatteryPower(containerId, timestamps, rbPower, rlPower, viewMode = "both", hybridPower = null) {
    const traces = [];
    const rlLabel = this.getRlLabel(viewMode);

    if (viewMode === "rb" || viewMode === "both") {
      traces.push({
        x: timestamps,
        y: rbPower,
        name: "Rule-Based (Baseline)",
        type: "scatter",
        mode: "lines",
        line: { color: "#f43f5e", width: 2, dash: "dot" },
      });
    }
    if (viewMode === "both" && hybridPower) {
      traces.push({
        x: timestamps,
        y: rlPower,
        name: "Forecast-Assisted PPO",
        type: "scatter",
        mode: "lines",
        line: { color: "#06b6d4", width: 2 },
      });
      traces.push({
        x: timestamps,
        y: hybridPower,
        name: "Hybrid DQN-PPO (Strategic)",
        type: "scatter",
        mode: "lines",
        line: { color: "#a855f7", width: 2.4 },
      });
    } else if (viewMode !== "rb") {
      traces.push({
        x: timestamps,
        y: rlPower,
        name: `${rlLabel} Battery Power (kW)`,
        type: "scatter",
        mode: "lines",
        line: { color: viewMode.includes("hybrid") ? "#a855f7" : "#06b6d4", width: 2.2 },
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: {
        ...darkLayoutDefaults.yaxis,
        title: "Battery Power (kW) [+ Discharging, - Charging]",
      },
    };

    this.plot(containerId, traces, layout);
  },

  renderBatterySOC(containerId, timestamps, rbSOC, rlSOC, minSOC = 20, maxSOC = 90, viewMode = "both", hybridSOC = null) {
    const traces = [];
    const rlLabel = this.getRlLabel(viewMode);

    if (viewMode === "rb" || viewMode === "both") {
      traces.push({
        x: timestamps,
        y: rbSOC.map(v => v * 100),
        name: "Rule-Based SOC (%)",
        type: "scatter",
        mode: "lines",
        line: { color: "#f43f5e", width: 2, dash: "dot" },
      });
    }
    if (viewMode === "both" && hybridSOC) {
      traces.push({
        x: timestamps,
        y: rlSOC.map(v => v * 100),
        name: "PPO SOC (%)",
        type: "scatter",
        mode: "lines",
        line: { color: "#06b6d4", width: 2 },
      });
      traces.push({
        x: timestamps,
        y: hybridSOC.map(v => v * 100),
        name: "Hybrid DQN-PPO SOC (%)",
        type: "scatter",
        mode: "lines",
        line: { color: "#a855f7", width: 2.5 },
      });
    } else if (viewMode !== "rb") {
      traces.push({
        x: timestamps,
        y: rlSOC.map(v => v * 100),
        name: `${rlLabel} SOC (%)`,
        type: "scatter",
        mode: "lines",
        line: { color: viewMode.includes("hybrid") ? "#a855f7" : "#06b6d4", width: 2.4 },
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: {
        ...darkLayoutDefaults.yaxis,
        title: "State of Charge (%)",
        range: [10, 100],
      },
      shapes: [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          y0: minSOC,
          y1: minSOC,
          line: { color: "#f87171", width: 1.5, dash: "dot" },
        },
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          y0: maxSOC,
          y1: maxSOC,
          line: { color: "#34d399", width: 1.5, dash: "dot" },
        },
      ],
      annotations: [
        { xref: "paper", x: 0.98, y: minSOC + 3, text: `Min Limit (${minSOC}%)`, showarrow: false, font: { color: "#f87171", size: 10 } },
        { xref: "paper", x: 0.98, y: maxSOC - 3, text: `Max Limit (${maxSOC}%)`, showarrow: false, font: { color: "#34d399", size: 10 } },
      ],
    };

    this.plot(containerId, traces, layout);
  },

  renderGridPower(containerId, timestamps, rbGrid, rlGrid, viewMode = "both", hybridGrid = null) {
    const traces = [];
    const rlLabel = this.getRlLabel(viewMode);

    if (viewMode === "rb" || viewMode === "both") {
      traces.push({
        x: timestamps,
        y: rbGrid,
        name: "Rule-Based Grid (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: "#f43f5e", width: 2, dash: "dot" },
      });
    }
    if (viewMode === "both" && hybridGrid) {
      traces.push({
        x: timestamps,
        y: rlGrid,
        name: "PPO Grid (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: "#06b6d4", width: 2 },
      });
      traces.push({
        x: timestamps,
        y: hybridGrid,
        name: "Hybrid DQN-PPO Grid (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: "#a855f7", width: 2.2 },
      });
    } else if (viewMode !== "rb") {
      traces.push({
        x: timestamps,
        y: rlGrid,
        name: `${rlLabel} Grid (kW)`,
        type: "scatter",
        mode: "lines",
        line: { color: viewMode.includes("hybrid") ? "#a855f7" : "#06b6d4", width: 2.2 },
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: { ...darkLayoutDefaults.yaxis, title: "Grid Power (kW)" },
    };

    this.plot(containerId, traces, layout);
  },

  renderTariffVsGrid(containerId, timestamps, tariff, rbGrid, rlGrid, viewMode = "both") {
    const traces = [
      {
        x: timestamps,
        y: tariff,
        name: "Tariff Rate (Rs./kWh)",
        type: "scatter",
        mode: "lines",
        line: { color: "#a855f7", width: 2.5, shape: "hv" },
        yaxis: "y1",
      }
    ];
    const rlLabel = this.getRlLabel(viewMode);

    if (viewMode !== "rb") {
      traces.push({
        x: timestamps,
        y: rlGrid,
        name: `${rlLabel} Grid Import (kW)`,
        type: "scatter",
        mode: "lines",
        line: { color: "#10b981", width: 2 },
        yaxis: "y2",
      });
    }

    if (viewMode === "rb" || viewMode === "both") {
      traces.push({
        x: timestamps,
        y: rbGrid,
        name: "RB Grid Import (kW)",
        type: "scatter",
        mode: "lines",
        line: { color: viewMode === "rb" ? "#38bdf8" : "#f87171", width: 1.8, dash: viewMode === "both" ? "dot" : "solid" },
        yaxis: "y2",
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: {
        ...darkLayoutDefaults.yaxis,
        title: "Tariff (Rs./kWh)",
        titlefont: { color: "#a855f7" },
        tickfont: { color: "#a855f7" },
      },
      yaxis2: {
        title: "Grid Import (kW)",
        titlefont: { color: "#10b981" },
        tickfont: { color: "#10b981" },
        overlaying: "y",
        side: "right",
        gridcolor: "rgba(0,0,0,0)",
      },
    };

    this.plot(containerId, traces, layout);
  },

  renderCumulativeCost(containerId, timestamps, rbCost, rlCost, viewMode = "both", hybridCost = null) {
    const traces = [];
    const rlLabel = this.getRlLabel(viewMode);

    if (viewMode === "rb" || viewMode === "both") {
      traces.push({
        x: timestamps,
        y: rbCost,
        name: "Rule-Based (Baseline)",
        type: "scatter",
        mode: "lines",
        line: { color: "#f43f5e", width: 2, dash: "dot" },
      });
    }
    if (viewMode === "both" && hybridCost) {
      traces.push({
        x: timestamps,
        y: rlCost,
        name: "Forecast-Assisted PPO",
        type: "scatter",
        mode: "lines",
        line: { color: "#06b6d4", width: 2 },
      });
      traces.push({
        x: timestamps,
        y: hybridCost,
        name: "Hybrid DQN-PPO (Strategic)",
        type: "scatter",
        mode: "lines",
        line: { color: "#a855f7", width: 2.5 },
      });
    } else if (viewMode !== "rb") {
      traces.push({
        x: timestamps,
        y: rlCost,
        name: `${rlLabel} Cumulative Cost (Rs.)`,
        type: "scatter",
        mode: "lines",
        line: { color: viewMode.includes("hybrid") ? "#a855f7" : "#06b6d4", width: 2.4 },
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "x unified",
      xaxis: {
        ...darkLayoutDefaults.xaxis,
        type: "date",
      },
      yaxis: { ...darkLayoutDefaults.yaxis, title: "Cumulative Cost (Rs.)" },
    };

    this.plot(containerId, traces, layout);
  },

  renderCostBar(containerId, rbCost, rlCost, viewMode = "both", hybridCost = null) {
    const rbVal = Number(rbCost) || 0;
    const rlVal = Number(rlCost) || 0;
    const hybVal = Number(hybridCost) || 0;
    const rlLabel = this.getRlLabel(viewMode);

    let xCategories = ["Rule-Based EMS (Baseline)", "Forecast-Assisted PPO"];
    let yValues = [rbVal, rlVal];
    let colors = ["#f43f5e", "#06b6d4"];
    let borderColors = ["#e11d48", "#0891b2"];
    let labels = [`Rs. ${rbVal.toFixed(2)}`, `Rs. ${rlVal.toFixed(2)}`];

    if (viewMode === "rb") {
      xCategories = ["Rule-Based EMS (Baseline)"];
      yValues = [rbVal];
      colors = ["#f43f5e"];
      borderColors = ["#e11d48"];
      labels = [`Rs. ${rbVal.toFixed(2)}`];
    } else if (viewMode === "ppo") {
      xCategories = ["Forecast-Assisted PPO"];
      yValues = [rlVal];
      colors = ["#06b6d4"];
      borderColors = ["#0891b2"];
      labels = [`Rs. ${rlVal.toFixed(2)}`];
    }

    const trace = {
      x: xCategories,
      y: yValues,
      type: "bar",
      marker: {
        color: colors,
        line: { color: borderColors, width: 1.5 },
      },
      text: labels,
      textposition: "auto",
      textfont: { color: "#ffffff", family: "'Outfit', 'Inter', sans-serif", size: 14, weight: 700 },
      width: xCategories.length > 2 ? [0.35, 0.35, 0.35] : [0.4],
    };

    const layout = {
      ...darkLayoutDefaults,
      hovermode: "closest",
      xaxis: {
        type: "category",
        tickfont: { color: "#f8fafc", size: 13, family: "'Outfit', 'Inter', sans-serif" },
        gridcolor: "rgba(255, 255, 255, 0.06)",
      },
      yaxis: {
        ...darkLayoutDefaults.yaxis,
        title: "Total Electricity Cost (Rs.)",
        rangemode: "tozero",
      },
    };

    this.plot(containerId, [trace], layout);
  },

  renderSecondaryMetrics(containerId, rbMetrics, ppoMetrics) {
    const categories = ["Grid Energy (kWh)", "Solar Util (%)", "Battery Thru (kWh)", "Peak Grid (kW)"];
    const rbVals = [
      rbMetrics.grid_energy_consumed_kwh || 0,
      rbMetrics.solar_utilization_pct || 0,
      rbMetrics.battery_throughput_kwh || 0,
      rbMetrics.peak_grid_demand_kw || 0,
    ];
    const ppoVals = [
      ppoMetrics.grid_energy_consumed_kwh || 0,
      ppoMetrics.solar_utilization_pct || 0,
      ppoMetrics.battery_throughput_kwh || 0,
      ppoMetrics.peak_grid_demand_kw || 0,
    ];

    const trace1 = {
      x: categories,
      y: rbVals,
      name: "Rule-Based EMS",
      type: "bar",
      marker: { color: "#64748b" },
    };

    const trace2 = {
      x: categories,
      y: ppoVals,
      name: "Forecast-PPO",
      type: "bar",
      marker: { color: "#06b6d4" },
    };

    const layout = {
      ...darkLayoutDefaults,
      barmode: "group",
      hovermode: "closest",
      xaxis: {
        type: "category",
        tickfont: { color: "#f8fafc", size: 12, family: "'Outfit', 'Inter', sans-serif" },
        gridcolor: "rgba(255, 255, 255, 0.06)",
      },
      yaxis: { ...darkLayoutDefaults.yaxis, title: "Value (Respective Units)" },
    };

    this.plot(containerId, [trace1, trace2], layout);
  },

  renderScenarioComparison(containerId, timeLabels, baseData, randomizedData, forecastData, labelBase, labelRand, baseColor, randColor, unit = "kW") {
    const traces = [
      {
        x: timeLabels,
        y: baseData,
        name: labelBase,
        type: "scatter",
        mode: "lines",
        line: { color: baseColor, width: 2, dash: "dot" },
      },
      {
        x: timeLabels,
        y: randomizedData,
        name: labelRand,
        type: "scatter",
        mode: "lines",
        line: { color: randColor, width: 2.5 },
        fill: "tozeroy",
        fillcolor: randColor.startsWith("#") ? `${randColor}18` : "rgba(56, 189, 248, 0.12)",
      },
    ];

    if (forecastData && forecastData.length > 0) {
      traces.push({
        x: timeLabels,
        y: forecastData,
        name: "Forecast Feature Input",
        type: "scatter",
        mode: "lines",
        line: { color: "#94a3b8", width: 1.5, dash: "dash" },
      });
    }

    const layout = {
      ...darkLayoutDefaults,
      xaxis: { ...darkLayoutDefaults.xaxis, title: "Time of Day (15-min intervals)" },
      yaxis: { ...darkLayoutDefaults.yaxis, title: `Power (${unit})` },
      hovermode: "x unified",
    };

    this.plot(containerId, traces, layout);
  },
};

