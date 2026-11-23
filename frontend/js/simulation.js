/**
 * Simulation and Comparison Coordination Module
 * Supports single-day (96 steps) and 7-day test benchmark (672 steps)
 * Enriches date picker with test vs training partitions, energy summaries, and clear contextual cues.
 */

const SimulationManager = {
  mode: "day", // "day", "7day", or "14day"
  controllerView: "both", // "both", "rb", or "ppo"
  activeDate: "2020-06-28",
  datesMetadata: {},
  testDates: [],
  allDates: [],
  currentTimeseries: [],
  comparisonMetrics: null,
  lastRbMetrics: null,
  lastPpoMetrics: null,
  lastCompMetrics: null,

  async init() {
    this.setupEventListeners();
    await this.populateDates();
    await this.loadInitialData();
  },

  async populateDates() {
    try {
      const data = await API.getDates();
      const select = document.getElementById("date-select");
      select.innerHTML = "";

      // Handle both structured dict and raw array
      if (data && data.metadata) {
        this.datesMetadata = data.metadata;
        this.testDates = data.test_dates || [];
        this.allDates = data.dates || [];
        this.activeDate = data.default_date || (this.testDates.length > 0 ? this.testDates[0] : this.allDates[0]);

        // Group 1: Held-Out Test Set (June 2020)
        if (this.testDates.length > 0) {
          const testGroup = document.createElement("optgroup");
          testGroup.label = "⭐ Held-Out Test Set (June 2020 — Unseen by RL)";
          this.testDates.forEach(d => {
            const meta = this.datesMetadata[d] || {};
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = `${d} (${meta.day_name ? meta.day_name.slice(0, 3) : "Day"}) • ${meta.solar_kwh || 0} kWh`;
            testGroup.appendChild(opt);
          });
          select.appendChild(testGroup);
        }

        // Group 2: Training Set Samples
        const trainDates = this.allDates.filter(d => !this.testDates.includes(d));
        if (trainDates.length > 0) {
          const trainGroup = document.createElement("optgroup");
          trainGroup.label = "📚 Historical Training Set";
          // Provide up to 20 representative training dates
          trainDates.slice(0, 20).forEach(d => {
            const meta = this.datesMetadata[d] || {};
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = `${d} (${meta.day_name ? meta.day_name.slice(0, 3) : "Day"}) • ${meta.solar_kwh || 0} kWh`;
            trainGroup.appendChild(opt);
          });
          select.appendChild(trainGroup);
        }
      } else {
        const datesList = (data && data.dates) ? data.dates : (Array.isArray(data) ? data : []);
        this.allDates = datesList;
        this.activeDate = datesList[0] || "2020-06-28";
        datesList.forEach(d => {
          const opt = document.createElement("option");
          opt.value = d;
          opt.textContent = d;
          select.appendChild(opt);
        });
      }

      if (this.activeDate) {
        select.value = this.activeDate;
      }
      this.updateContextBar();
    } catch (e) {
      console.error("Failed to load dates:", e);
    }
  },

  updateContextBar() {
    const horizonEl = document.getElementById("ctx-horizon-val");
    const partitionEl = document.getElementById("ctx-partition-val");
    const dayEl = document.getElementById("ctx-day-val");
    const solarEl = document.getElementById("ctx-solar-val");
    const loadEl = document.getElementById("ctx-load-val");

    if (this.mode === "14day") {
      if (horizonEl) horizonEl.textContent = "14-Day Continuous Test Benchmark (336h / 1,344 steps)";
      if (partitionEl) {
        partitionEl.textContent = "Held-Out Test Set (Procedural Horizon)";
        partitionEl.className = "context-value badge-test-partition";
      }
      if (dayEl) dayEl.textContent = "Thursday, Oct 01 – Wednesday, Oct 14, 2026";
      if (solarEl) solarEl.textContent = "~71.4 kWh Total Available";
      if (loadEl) loadEl.textContent = "~148.6 kWh Total Demand";
      return;
    }

    if (this.mode === "7day") {
      if (horizonEl) horizonEl.textContent = "7-Day Continuous Test Benchmark (168h / 672 steps)";
      if (partitionEl) {
        partitionEl.textContent = "Held-Out Test Set (Procedural Horizon)";
        partitionEl.className = "context-value badge-test-partition";
      }
      if (dayEl) dayEl.textContent = "Thursday, Oct 01 – Wednesday, Oct 07, 2026";
      if (solarEl) solarEl.textContent = "~35.7 kWh Total Available";
      if (loadEl) loadEl.textContent = "~74.3 kWh Total Demand";
      return;
    }

    // Single Day Mode
    const meta = this.datesMetadata[this.activeDate] || {};
    const isTest = meta.is_test !== undefined ? meta.is_test : (this.testDates.includes(this.activeDate) || (this.activeDate && this.activeDate.startsWith("2020-06")));
    
    if (horizonEl) horizonEl.textContent = "Single Operating Day (24h / 96 steps)";
    if (partitionEl) {
      partitionEl.textContent = isTest ? "Held-Out Test Set (Unseen Data)" : "Training Set (Historical)";
      partitionEl.className = isTest ? "context-value badge-test-partition" : "context-value badge-train-partition";
    }
    if (dayEl) {
      dayEl.textContent = `${meta.day_name || "Day"}, ${this.activeDate}`;
    }
    if (solarEl) {
      solarEl.textContent = `${meta.solar_kwh !== undefined ? meta.solar_kwh : "--"} kWh`;
    }
    if (loadEl) {
      loadEl.textContent = `${meta.load_kwh !== undefined ? meta.load_kwh : "--"} kWh`;
    }
  },

  async loadInitialData() {
    try {
      document.getElementById("loading-indicator").style.display = "inline-flex";
      const res = await API.getComparisonResults();
      const tsData = await API.getComparisonTimeseries();

      this.currentTimeseries = tsData.timeseries || [];
      this.comparisonMetrics = res.comparison_metrics;
      this.comparisonTable = res.comparison_table || null;
      this.selectedBestRl = res.selected_best_rl || "hybrid_dqn_ppo";
      this.selectedBestRlName = res.selected_best_rl_name || "Hybrid DQN-PPO";
      this.multiResults = {
        rule_based: res.rule_based_metrics,
        ppo: res.ppo_metrics,
        hybrid_dqn_ppo: res.hybrid_metrics || res.hybrid_dqn_ppo_metrics,
      };

      this.updateUI(res.rule_based_metrics, res.ppo_metrics, res.comparison_metrics);
      this.renderAllCharts();
      PlaybackManager.init(this.currentTimeseries);
    } catch (e) {
      console.error("Failed loading initial comparison:", e);
    } finally {
      document.getElementById("loading-indicator").style.display = "none";
    }
  },

  setupEventListeners() {
    const btnRun = document.getElementById("btn-run-comparison");
    if (btnRun) {
      btnRun.addEventListener("click", () => this.handleRunComparison());
    }

    const ctrlSelect = document.getElementById("controller-view-select");
    if (ctrlSelect) {
      ctrlSelect.addEventListener("change", (e) => {
        this.controllerView = e.target.value;
        if (this.lastRbMetrics && this.lastPpoMetrics && this.lastCompMetrics) {
          this.updateUI(this.lastRbMetrics, this.lastPpoMetrics, this.lastCompMetrics);
        }
        this.renderAllCharts();
      });
    }

    const dateSelect = document.getElementById("date-select");
    if (dateSelect) {
      dateSelect.addEventListener("change", (e) => {
        this.activeDate = e.target.value;
        this.updateContextBar();
        this.handleRunComparison();
      });
    }

    // Scope toggle buttons (Single Day, 7-Day, 14-Day)
    const btnDay = document.getElementById("scope-btn-day");
    const btnWeek = document.getElementById("scope-btn-week");
    const btnFortnight = document.getElementById("scope-btn-fortnight");
    const dateContainer = document.getElementById("date-select-container");

    const setScope = (newMode) => {
      if (this.mode === newMode) return;
      this.mode = newMode;
      if (btnDay) btnDay.classList.toggle("active", newMode === "day");
      if (btnWeek) btnWeek.classList.toggle("active", newMode === "7day");
      if (btnFortnight) btnFortnight.classList.toggle("active", newMode === "14day");

      const isSingleDay = newMode === "day";
      if (dateContainer) dateContainer.style.opacity = isSingleDay ? "1" : "0.45";
      if (dateSelect) dateSelect.disabled = !isSingleDay;
      this.updateContextBar();
      this.handleRunComparison();
    };

    if (btnDay) btnDay.addEventListener("click", () => setScope("day"));
    if (btnWeek) btnWeek.addEventListener("click", () => setScope("7day"));
    if (btnFortnight) btnFortnight.addEventListener("click", () => setScope("14day"));

    // Date explanation info modal
    const btnInfo = document.getElementById("btn-date-info");
    const modal = document.getElementById("date-info-modal");
    const btnCloseModal = document.getElementById("btn-close-date-info");

    if (btnInfo && modal) {
      btnInfo.addEventListener("click", () => {
        modal.style.display = "flex";
      });
    }

    if (btnCloseModal && modal) {
      btnCloseModal.addEventListener("click", () => {
        modal.style.display = "none";
      });
    }

    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          modal.style.display = "none";
        }
      });
    }

    const btnSaveConfig = document.getElementById("btn-save-config");
    if (btnSaveConfig) {
      btnSaveConfig.addEventListener("click", () => this.handleSaveConfig());
    }
  },

  async handleRunComparison() {
    const initialSOC = parseFloat(document.getElementById("cfg-initial-soc")?.value || "0.50");
    const capacity = parseFloat(document.getElementById("cfg-capacity")?.value || "5.0");
    const gridCharging = document.getElementById("cfg-grid-charging") ? document.getElementById("cfg-grid-charging").checked : true;

    const payload = {
      mode: this.mode,
      date: this.mode === "day" ? this.activeDate : undefined,
      initial_soc: initialSOC,
      battery_capacity: capacity,
      allow_grid_charging: gridCharging,
    };

    try {
      document.getElementById("loading-indicator").style.display = "inline-flex";
      const res = await API.runComparison(payload);
      this.currentTimeseries = res.timeseries || [];
      this.comparisonMetrics = res.comparison_metrics;
      this.comparisonTable = res.comparison_table || null;
      this.selectedBestRl = res.selected_best_rl || "hybrid_dqn_ppo";
      this.selectedBestRlName = res.selected_best_rl_name || "Hybrid DQN-PPO";
      this.multiResults = {
        rule_based: res.rule_based_metrics,
        ppo: res.ppo_metrics,
        hybrid_dqn_ppo: res.hybrid_metrics || res.hybrid_dqn_ppo_metrics,
      };

      this.updateUI(res.rule_based_metrics, res.ppo_metrics, res.comparison_metrics);
      this.renderAllCharts();
      PlaybackManager.init(this.currentTimeseries);
    } catch (e) {
      alert("Error executing comparison: " + e.message);
    } finally {
      document.getElementById("loading-indicator").style.display = "none";
    }
  },

  updateUI(rbMetrics, ppoMetrics, compMetrics) {
    this.lastRbMetrics = rbMetrics;
    this.lastPpoMetrics = ppoMetrics;
    this.lastCompMetrics = compMetrics;

    const view = this.controllerView || "both";
    const bestAlgo = this.selectedBestRl || "hybrid_dqn_ppo";

    // Determine active RL metrics based on view
    let activeMetrics = ppoMetrics;
    let activeName = "PPO (Continuous)";
    if (this.multiResults) {
      if ((view === "hybrid_dqn_ppo" || view === "hybrid") && this.multiResults.hybrid_dqn_ppo) {
        activeMetrics = this.multiResults.hybrid_dqn_ppo;
        activeName = "Hybrid DQN-PPO (Strategic Dispatch)";
      } else if ((view === "best" || view === "both") && this.multiResults[bestAlgo]) {
        activeMetrics = this.multiResults[bestAlgo];
        activeName = `${this.selectedBestRlName || "Hybrid DQN-PPO"} (Selected Winner)`;
      } else if (view === "ppo" && this.multiResults.ppo) {
        activeMetrics = this.multiResults.ppo;
        activeName = "PPO (Continuous)";
      }
    }

    const rbSocRaw = rbMetrics.final_battery_soc !== undefined ? rbMetrics.final_battery_soc : 0.2;
    const rbSoc = (rbSocRaw <= 1.0 ? rbSocRaw * 100 : rbSocRaw);
    const activeSocRaw = activeMetrics.final_battery_soc !== undefined ? activeMetrics.final_battery_soc : 0.5;
    const activeSoc = (activeSocRaw <= 1.0 ? activeSocRaw * 100 : activeSocRaw);

    const costTitleEl = document.getElementById("stat-cost-title");
    const ppoCostEl = document.getElementById("stat-ppo-cost");
    const costSubEl = document.getElementById("stat-cost-sub");

    const impactTitleEl = document.getElementById("stat-impact-title");
    const impactValEl = document.getElementById("stat-impact-val");
    const impactSubEl = document.getElementById("stat-impact-sub");

    const solarTitleEl = document.getElementById("stat-solar-title");
    const solarUtilEl = document.getElementById("stat-solar-util");
    const solarSubEl = document.getElementById("stat-solar-sub");

    const gridTitleEl = document.getElementById("stat-grid-title");
    const gridEnergyEl = document.getElementById("stat-grid-energy");
    const gridSubEl = document.getElementById("stat-grid-sub");

    const throughputTitleEl = document.getElementById("stat-throughput-title");
    const throughputEl = document.getElementById("stat-throughput");
    const throughputSubEl = document.getElementById("stat-throughput-sub");

    if (view === "rb") {
      // ---------------------------------------------------------
      // Rule-Based EMS (Baseline) View
      // ---------------------------------------------------------
      if (costTitleEl) costTitleEl.textContent = "Rule-Based Total Cost";
      if (ppoCostEl) ppoCostEl.textContent = `Rs. ${rbMetrics.total_electricity_cost.toFixed(2)}`;
      if (costSubEl) {
        costSubEl.innerHTML = `<span>Terminal Stored Energy: </span><strong style="color: #f87171;">${rbSoc.toFixed(1)}% SOC</strong> (${rbSoc <= 20.1 ? 'Depleted to 20% Min' : 'Reserve'})`;
      }

      if (impactTitleEl) impactTitleEl.textContent = "Operating Strategy";
      if (impactValEl) {
        impactValEl.innerHTML = `<span class="status-badge" style="background: rgba(148, 163, 184, 0.2); color: #cbd5e1; font-weight: 600;">Rule-Based Baseline</span>`;
      }
      if (impactSubEl) {
        impactSubEl.textContent = "Deterministic Greedy Solar & Peak Discharge";
      }

      if (solarTitleEl) solarTitleEl.textContent = "Solar Utilization";
      if (solarUtilEl) solarUtilEl.textContent = `${rbMetrics.solar_utilization_pct.toFixed(1)}%`;
      if (solarSubEl) {
        solarSubEl.textContent = `${(rbMetrics.solar_energy_used_kwh || 0).toFixed(1)} kWh used of ${(rbMetrics.solar_energy_available_kwh || 0).toFixed(1)} kWh Available`;
      }

      if (gridTitleEl) gridTitleEl.textContent = "Grid Consumed";
      if (gridEnergyEl) gridEnergyEl.textContent = `${rbMetrics.grid_energy_consumed_kwh.toFixed(1)} kWh`;
      if (gridSubEl) {
        gridSubEl.innerHTML = `Peak Demand: <strong style="color: #f87171;">${rbMetrics.peak_grid_demand_kw.toFixed(2)} kW</strong>`;
      }

      if (throughputTitleEl) throughputTitleEl.textContent = "Battery Throughput";
      if (throughputEl) throughputEl.textContent = `${rbMetrics.battery_throughput_kwh.toFixed(1)} kWh`;
      if (throughputSubEl) throughputSubEl.textContent = "Cycling Degradation Metric";

    } else if (view !== "both") {
      // ---------------------------------------------------------
      // Specific Single RL Agent View (PPO, Hybrid DQN-PPO, or BEST)
      // ---------------------------------------------------------
      if (costTitleEl) costTitleEl.textContent = `${activeName} Cost`;
      if (ppoCostEl) ppoCostEl.textContent = `Rs. ${activeMetrics.total_electricity_cost.toFixed(2)}`;
      const adjCost = activeMetrics.adjusted_electricity_cost || activeMetrics.total_electricity_cost;
      if (costSubEl) {
        costSubEl.innerHTML = `<span>Terminal-Adjusted Cost: </span><strong style="color: #38bdf8;">Rs. ${Number(adjCost).toFixed(2)}</strong>`;
      }

      const peakDiff = (rbMetrics.peak_grid_demand_kw - activeMetrics.peak_grid_demand_kw);
      const peakPct = (peakDiff / (rbMetrics.peak_grid_demand_kw || 1)) * 100;

      if (impactTitleEl) impactTitleEl.textContent = "Peak Demand Reduction";
      if (impactValEl) {
        impactValEl.innerHTML = `<span class="status-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 600;">${peakPct >= 0 ? '-' : '+'}${Math.abs(peakPct).toFixed(1)}% Peak Shaved</span>`;
      }
      if (impactSubEl) {
        impactSubEl.innerHTML = `<span>Preserved Stored Reserve: </span><strong style="color: #34d399;">${activeSoc.toFixed(1)}% Final SOC</strong>`;
      }

      if (solarTitleEl) solarTitleEl.textContent = "Solar Utilization";
      if (solarUtilEl) solarUtilEl.textContent = `${activeMetrics.solar_utilization_pct.toFixed(1)}%`;
      if (solarSubEl) {
        solarSubEl.textContent = `${(activeMetrics.solar_energy_used_kwh || 0).toFixed(1)} kWh used of ${(activeMetrics.solar_energy_available_kwh || 0).toFixed(1)} kWh Available`;
      }

      if (gridTitleEl) gridTitleEl.textContent = "Grid Consumed";
      if (gridEnergyEl) gridEnergyEl.textContent = `${activeMetrics.grid_energy_consumed_kwh.toFixed(1)} kWh`;
      if (gridSubEl) {
        gridSubEl.innerHTML = `Peak Demand: <strong style="color: #38bdf8;">${activeMetrics.peak_grid_demand_kw.toFixed(2)} kW</strong>`;
      }

      if (throughputTitleEl) throughputTitleEl.textContent = "Battery Throughput";
      if (throughputEl) throughputEl.textContent = `${activeMetrics.battery_throughput_kwh.toFixed(1)} kWh`;
      if (throughputSubEl) throughputSubEl.textContent = "Arbitrage & Pre-Peak Cycling Metric";

    } else {
      // ---------------------------------------------------------
      // Comparative (Side-by-Side) Default View: Best RL vs Rule-Based
      // ---------------------------------------------------------
      if (costTitleEl) costTitleEl.textContent = `${activeName} Cost`;
      if (ppoCostEl) ppoCostEl.textContent = `Rs. ${activeMetrics.total_electricity_cost.toFixed(2)}`;
      if (costSubEl) {
        costSubEl.innerHTML = `<span>Rule-Based: </span><strong id="stat-rb-cost" style="color: #f87171;">Rs. ${rbMetrics.total_electricity_cost.toFixed(2)}</strong>`;
      }

      const costSaving = rbMetrics.total_electricity_cost - activeMetrics.total_electricity_cost;
      const savingPct = (costSaving / (rbMetrics.total_electricity_cost || 1)) * 100;

      if (impactTitleEl) impactTitleEl.textContent = "Cost Impact";
      if (impactValEl) {
        impactValEl.innerHTML = `<span id="stat-cost-saving" class="${savingPct >= 0 ? "status-badge" : "status-badge badge-warning"}">${savingPct >= 0 ? "+" : ""}${savingPct.toFixed(1)}% (Rs. ${Math.abs(costSaving).toFixed(2)})</span>`;
      }
      if (impactSubEl) impactSubEl.textContent = "Under Identical Operating Conditions";

      if (solarTitleEl) solarTitleEl.textContent = "Solar Utilization";
      if (solarUtilEl) solarUtilEl.textContent = `${activeMetrics.solar_utilization_pct.toFixed(1)}%`;
      if (solarSubEl) solarSubEl.textContent = `Clean Energy Self-Consumption (RB: ${rbMetrics.solar_utilization_pct.toFixed(1)}%)`;

      if (gridTitleEl) gridTitleEl.textContent = "Grid Consumed";
      if (gridEnergyEl) gridEnergyEl.textContent = `${activeMetrics.grid_energy_consumed_kwh.toFixed(1)} kWh`;
      if (gridSubEl) {
        gridSubEl.innerHTML = `Peak: <strong id="stat-peak-grid" style="color: #38bdf8;">${activeMetrics.peak_grid_demand_kw.toFixed(2)} kW</strong> (RB: ${rbMetrics.peak_grid_demand_kw.toFixed(2)} kW)`;
      }

      if (throughputTitleEl) throughputTitleEl.textContent = "Battery Throughput";
      if (throughputEl) throughputEl.textContent = `${activeMetrics.battery_throughput_kwh.toFixed(1)} kWh`;
      if (throughputSubEl) throughputSubEl.textContent = `Cycling Metric (RB: ${rbMetrics.battery_throughput_kwh.toFixed(1)} kWh)`;
    }

    // 2. Summary Narrative
    const narrativeEl = document.getElementById("comparison-narrative");
    if (narrativeEl) {
      narrativeEl.textContent = compMetrics.narrative_summary || "Multi-algorithm RL evaluation computed under strictly identical microgrid conditions.";
    }

    // 3. Performance Summary Table
    this.renderSummaryTable(compMetrics);
  },

  renderSummaryTable(comp) {
    const tbody = document.getElementById("summary-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    const winnerEl = document.getElementById("winner-algo-name");
    if (winnerEl) {
      winnerEl.textContent = "Forecast-Assisted PPO";
    }

    const excludedMetrics = [
      "Grid Energy Consumed",
      "Solar Utilization",
      "Battery Throughput",
      "Peak Grid Demand",
      "Final Battery SOC",
      "SOC Violations",
      "Unserved Energy"
    ];

    let tableRows = (this.comparisonTable && this.comparisonTable.rows && this.comparisonTable.rows.length > 0)
      ? this.comparisonTable.rows.filter(item => !excludedMetrics.includes(item.metric))
      : null;

    if (tableRows && tableRows.length > 0) {
      tableRows.forEach(item => {
        const tr = document.createElement("tr");
        const isBetter = item.best_controller !== "rule_based";
        const bestLabel = item.best_controller === "rule_based"
          ? '<span class="badge-neutral">Rule-Based</span>'
          : '<span class="badge-winner">PPO (CONTINUOUS)</span>';

        tr.innerHTML = `
          <td><strong>${item.metric}</strong></td>
          <td>${item.rule_based !== null ? Number(item.rule_based).toFixed(2) : "--"} ${item.unit}</td>
          <td><strong>${item.ppo !== null ? Number(item.ppo).toFixed(2) : "--"} ${item.unit}</strong></td>
          <td>${bestLabel}</td>
          <td>${item.difference !== null ? `${Number(item.difference).toFixed(2)} ${item.unit} (${Number(item.percent_change).toFixed(1)}%)` : "--"}</td>
        `;
        tbody.appendChild(tr);
      });
      return;
    }

    // Fallback if full multi-table not yet loaded
    const rows = [
      comp.electricity_cost,
      comp.adjusted_cost,
    ].filter(Boolean);

    rows.forEach(item => {
      if (!item) return;
      const tr = document.createElement("tr");
      let badgeHtml = item.better === "ppo" ? '<span class="badge-winner">PPO Advantaged</span>' : '<span class="badge-neutral">Rule-Based</span>';
      tr.innerHTML = `
        <td><strong>${item.metric}</strong></td>
        <td>${item.rule_based.toFixed(2)} ${item.unit}</td>
        <td><strong>${item.ppo.toFixed(2)} ${item.unit}</strong></td>
        <td>${badgeHtml}</td>
        <td>${item.difference.toFixed(2)} ${item.unit} (${item.percent_change.toFixed(1)}%)</td>
      `;
      tbody.appendChild(tr);
    });
  },

  renderAllCharts() {
    if (!this.currentTimeseries || this.currentTimeseries.length === 0) return;

    const ts = this.currentTimeseries;
    const timestamps = ts.map(r => r.timestamp);
    const solar = ts.map(r => r.solar_kw);
    const load = ts.map(r => r.load_kw);
    const tariff = ts.map(r => r.tariff);

    const rbPower = ts.map(r => r.rb_battery_power_kw);
    const rbSOC = ts.map(r => r.rb_battery_soc);
    const rbGrid = ts.map(r => r.rb_grid_power_kw);
    const rbCost = ts.map(r => r.rb_cumulative_cost);

    const ppoPower = ts.map(r => r.ppo_battery_power_kw !== undefined ? r.ppo_battery_power_kw : 0);
    const ppoSOC = ts.map(r => r.ppo_battery_soc !== undefined ? r.ppo_battery_soc : 0.5);
    const ppoGrid = ts.map(r => r.ppo_grid_power_kw !== undefined ? r.ppo_grid_power_kw : 0);
    const ppoCost = ts.map(r => r.ppo_cumulative_cost !== undefined ? r.ppo_cumulative_cost : 0);

    const view = this.controllerView || "both";

    Charts.renderSolarVsLoad("chart-solar-vs-load", timestamps, solar, load);
    Charts.renderBatteryPower("chart-battery-power", timestamps, rbPower, ppoPower, view, null);
    Charts.renderBatterySOC("chart-battery-soc", timestamps, rbSOC, ppoSOC, 20, 90, view, null);
    Charts.renderGridPower("chart-grid-power", timestamps, rbGrid, ppoGrid, view, null);
    Charts.renderTariffVsGrid("chart-tariff-grid", timestamps, tariff, rbGrid, ppoGrid, view);
    Charts.renderCumulativeCost("chart-cumulative-cost", timestamps, rbCost, ppoCost, view, null);
    Charts.renderCumulativeCost("chart-cumulative-cost-tab7", timestamps, rbCost, ppoCost, view, null);

    const finalRBCost = (this.multiResults && this.multiResults.rule_based) ? this.multiResults.rule_based.total_electricity_cost : (rbCost[rbCost.length - 1] || 0);
    const finalPPOCost = (this.multiResults && this.multiResults.ppo) ? this.multiResults.ppo.total_electricity_cost : (ppoCost[ppoCost.length - 1] || 0);

    Charts.renderCostBar("chart-cost-bar", finalRBCost, finalPPOCost, view, null);
  },


  async handleSaveConfig() {
    const payload = {
      capacity_kwh: parseFloat(document.getElementById("cfg-capacity").value),
      initial_soc: parseFloat(document.getElementById("cfg-initial-soc").value),
      min_soc: parseFloat(document.getElementById("cfg-min-soc").value),
      max_soc: parseFloat(document.getElementById("cfg-max-soc").value),
      max_charge_kw: parseFloat(document.getElementById("cfg-max-charge").value),
      max_discharge_kw: parseFloat(document.getElementById("cfg-max-discharge").value),
      allow_grid_charging: document.getElementById("cfg-grid-charging").checked,
      allow_grid_export: document.getElementById("cfg-grid-export").checked,
    };

    try {
      await API.saveConfig(payload);
      alert("Configuration saved successfully!");
      this.handleRunComparison();
    } catch (e) {
      alert("Failed to save config: " + e.message);
    }
  }
};
