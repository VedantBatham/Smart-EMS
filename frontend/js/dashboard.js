/**
 * Dashboard Navigation, Full-Day Playback, Decision Explorer, and Energy Flow Animation
 */

const PlaybackManager = {
  timeseries: [],
  currentIndex: 0,
  isPlaying: false,
  timer: null,
  speedMs: 400,

  init(timeseries) {
    this.timeseries = timeseries || [];
    this.currentIndex = 0;
    this.pause();

    const slider = document.getElementById("playback-slider");
    if (slider) {
      slider.max = Math.max(0, this.timeseries.length - 1);
      slider.value = 0;
      slider.addEventListener("input", (e) => {
        this.currentIndex = parseInt(e.target.value);
        this.renderStep(this.currentIndex);
      });
    }

    const btnPlay = document.getElementById("btn-play");
    if (btnPlay) {
      btnPlay.addEventListener("click", () => this.togglePlay());
    }

    const btnPrev = document.getElementById("btn-prev");
    if (btnPrev) {
      btnPrev.addEventListener("click", () => this.prevStep());
    }

    const btnNext = document.getElementById("btn-next");
    if (btnNext) {
      btnNext.addEventListener("click", () => this.nextStep());
    }

    const btnRestart = document.getElementById("btn-restart");
    if (btnRestart) {
      btnRestart.addEventListener("click", () => {
        this.currentIndex = 0;
        this.renderStep(0);
      });
    }

    const speedSelect = document.getElementById("playback-speed");
    if (speedSelect) {
      speedSelect.addEventListener("change", (e) => {
        this.speedMs = parseInt(e.target.value);
        if (this.isPlaying) {
          this.pause();
          this.play();
        }
      });
    }

    if (this.timeseries.length > 0) {
      this.renderStep(0);
    }
  },

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  },

  play() {
    this.isPlaying = true;
    const playBtn = document.getElementById("btn-play");
    if (playBtn) playBtn.innerHTML = "&#10074;&#10074;"; // Pause icon

    this.timer = setInterval(() => {
      if (this.currentIndex >= this.timeseries.length - 1) {
        this.pause();
      } else {
        this.nextStep();
      }
    }, this.speedMs);
  },

  pause() {
    this.isPlaying = false;
    if (this.timer) clearInterval(this.timer);
    const playBtn = document.getElementById("btn-play");
    if (playBtn) playBtn.innerHTML = "&#9654;"; // Play icon
  },

  nextStep() {
    if (this.currentIndex < this.timeseries.length - 1) {
      this.currentIndex++;
      this.renderStep(this.currentIndex);
    }
  },

  prevStep() {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      this.renderStep(this.currentIndex);
    }
  },

  renderStep(index) {
    if (!this.timeseries || !this.timeseries[index]) return;

    const row = this.timeseries[index];
    const slider = document.getElementById("playback-slider");
    if (slider) slider.value = index;

    const safeSet = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };

    safeSet("playback-timestamp", row.timestamp || "--:--");

    // Update Live Flow Node Values
    safeSet("flow-val-solar", `${(row.solar_kw || 0).toFixed(2)} kW`);
    safeSet("flow-val-load", `${(row.load_kw || 0).toFixed(2)} kW`);
    safeSet("flow-val-batt", `${Math.abs(row.ppo_battery_power_kw || 0).toFixed(2)} kW`);
    safeSet("flow-val-grid", `${(row.ppo_grid_power_kw || 0).toFixed(2)} kW`);

    // Battery SOC Gauge
    const socPct = ((row.ppo_battery_soc || 0) * 100).toFixed(1);
    const gaugeEl = document.getElementById("playback-soc-bar");
    if (gaugeEl) gaugeEl.style.width = `${socPct}%`;
    safeSet("playback-soc-text", `${socPct}%`);

    // Active Controller & Tariff
    safeSet("playback-tariff", `Rs. ${(row.tariff || 0).toFixed(1)} / kWh`);
    safeSet("playback-cost", `Rs. ${(row.ppo_interval_cost || 0).toFixed(2)}`);

    // Decision Explorer Updates
    safeSet("exp-timestamp", row.timestamp || "--:--");
    safeSet("exp-load", `${(row.load_kw || 0).toFixed(2)} kW`);
    safeSet("exp-solar", `${(row.solar_kw || 0).toFixed(2)} kW`);
    safeSet("exp-soc", `${socPct}%`);
    safeSet("exp-tariff", `Rs. ${(row.tariff || 0).toFixed(1)}/kWh`);

    // Rule-Based Decision Display
    safeSet("exp-rb-action", `${row.rb_action || "Idle"} (${(row.rb_battery_power_kw || 0).toFixed(2)} kW)`);
    safeSet("exp-rb-grid", `${(row.rb_grid_power_kw || 0).toFixed(2)} kW`);
    safeSet("exp-rb-cost", `Rs. ${(row.rb_interval_cost || 0).toFixed(2)}`);
    safeSet("exp-rb-desc", row.rb_explanation || "Reactive rule dispatch based on immediate balance.");

    // PPO Decision Display
    safeSet("exp-ppo-action", `${row.ppo_action || "Hold"} (${(row.ppo_battery_power_kw || 0).toFixed(2)} kW)`);
    safeSet("exp-ppo-grid", `${(row.ppo_grid_power_kw || 0).toFixed(2)} kW`);
    safeSet("exp-ppo-cost", `Rs. ${(row.ppo_interval_cost || 0).toFixed(2)}`);
    safeSet("exp-ppo-desc", row.ppo_explanation || "PPO neural policy evaluated forward tariff and load.");

    // Hybrid DQN-PPO Decision Display
    const hybAction = row.hybrid_dqn_ppo_action || row.hybrid_action || row.ppo_action || "Strategic Hold";
    const hybPower = row.hybrid_dqn_ppo_battery_power_kw !== undefined ? row.hybrid_dqn_ppo_battery_power_kw : (row.ppo_battery_power_kw || 0);
    const hybGrid = row.hybrid_dqn_ppo_grid_power_kw !== undefined ? row.hybrid_dqn_ppo_grid_power_kw : (row.ppo_grid_power_kw || 0);
    const hybCost = (row.tariff || 0) * hybGrid * 0.25;
    const hybDesc = row.hybrid_dqn_ppo_explanation || row.hybrid_explanation || row.ppo_explanation || "DQN macro-regime selected and modulated via continuous PPO policy.";

    safeSet("exp-hyb-action", `${hybAction} (${hybPower.toFixed(2)} kW)`);
    safeSet("exp-hyb-grid", `${hybGrid.toFixed(2)} kW`);
    safeSet("exp-hyb-cost", `Rs. ${hybCost.toFixed(2)}`);
    safeSet("exp-hyb-desc", hybDesc);

    // Update Energy Flow Paths (SVG active classes)
    this.updateFlowPaths(row);
  },

  updateFlowPaths(row) {
    const pSolarToLoad = document.getElementById("path-solar-load");
    const pSolarToBatt = document.getElementById("path-solar-batt");
    const pBattToLoad = document.getElementById("path-batt-load");
    const pGridToLoad = document.getElementById("path-grid-load");
    const pGridToBatt = document.getElementById("path-grid-batt");

    const solar = row.solar_kw || 0;
    const load = row.load_kw || 0;
    const battP = row.ppo_battery_power_kw || 0;
    const gridP = row.ppo_grid_power_kw || 0;

    // Reset
    [pSolarToLoad, pSolarToBatt, pBattToLoad, pGridToLoad, pGridToBatt].forEach(p => {
      if (p) p.setAttribute("class", "flow-path");
    });

    if (solar > 0.05) {
      if (pSolarToLoad) pSolarToLoad.setAttribute("class", "flow-path active solar-active");
      if (battP < -0.05 && solar > load) {
        if (pSolarToBatt) pSolarToBatt.setAttribute("class", "flow-path active solar-active");
      }
    }

    if (battP > 0.05 && pBattToLoad) {
      pBattToLoad.setAttribute("class", "flow-path active battery-active");
    }

    if (gridP > 0.05) {
      if (pGridToLoad) pGridToLoad.setAttribute("class", "flow-path active grid-active");
      if (battP < -0.05 && solar <= load && pGridToBatt) {
        pGridToBatt.setAttribute("class", "flow-path active grid-active");
      }
    }
  }
};

const ScenarioManager = {
  currentData: null,

  async init() {
    const btnSample = document.getElementById("btn-sample-scenario");
    if (btnSample) {
      btnSample.addEventListener("click", () => this.loadScenario());
    }
    await this.loadScenario();
  },

  async loadScenario(dayIdx = null) {
    try {
      const data = await API.getScenarioSample(dayIdx);
      if (!data || !data.time_labels) return;
      this.currentData = data;
      this.renderMetadata(data);
      this.renderCharts(data);
    } catch (e) {
      console.error("Failed to load scenario sample:", e);
    }
  },

  renderMetadata(data) {
    const safeSet = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    };

    safeSet("scen-base-date", data.base_date || "Historical Day");
    const meta = data.metadata || {};
    safeSet("scen-solar-scale", `k_s = ${meta.solar_scaling_k_s !== undefined ? meta.solar_scaling_k_s : "1.0"}`);
    const weather = meta.weather_event ? meta.weather_event.replace(/_/g, " ").toUpperCase() : "CLEAR SKY";
    safeSet("scen-weather-event", `Event: ${weather}`);

    safeSet("scen-load-scale", `k_l = ${meta.load_scaling_k_l !== undefined ? meta.load_scaling_k_l : "1.0"}`);
    const shift = meta.peak_shift_steps ? `${meta.peak_shift_steps > 0 ? "+" : ""}${meta.peak_shift_steps * 15} min` : "0 min";
    const spikes = meta.demand_spikes_injected ? ` (${meta.demand_spikes_injected} spikes)` : "";
    safeSet("scen-load-shift", `Shift: ${shift}${spikes}`);

    safeSet("scen-initial-soc", `${((data.initial_soc || 0.5) * 100).toFixed(1)}%`);
    if (data.methodology_statement) {
      safeSet("scenario-methodology-text", `"${data.methodology_statement}"`);
    }
  },

  renderCharts(data) {
    if (!window.Charts || !window.Charts.renderScenarioComparison) return;

    // 1. Solar Profile Comparison
    Charts.renderScenarioComparison(
      "chart-scenario-solar",
      data.time_labels,
      data.base_solar_kw || data.base_solar,
      data.randomized_solar_kw || data.randomized_solar,
      data.future_solar_1h_avg,
      "Base Historical Solar",
      "Randomized Training Solar",
      "#ca8a04",
      "#f59e0b",
      "kW"
    );

    // 2. Load Profile Comparison
    Charts.renderScenarioComparison(
      "chart-scenario-load",
      data.time_labels,
      data.base_load_kw || data.base_load,
      data.randomized_load_kw || data.randomized_load,
      data.future_load_1h_avg,
      "Base Historical Load",
      "Randomized Training Load",
      "#64748b",
      "#06b6d4",
      "kW"
    );
  }
};

const Dashboard = {
  async init() {
    this.setupSidebarToggle();
    this.setupTabNavigation();
    
    // 1. Initialize core simulation and overview charts first
    try {
      await SimulationManager.init();
    } catch (e) {
      console.error("Failed to initialize SimulationManager:", e);
    }

    // 2. Load ML forecast metrics and charts safely
    try {
      await this.loadForecastViews();
    } catch (e) {
      console.error("Failed to load forecast views:", e);
    }
  },

  setupSidebarToggle() {
    const toggleBtn = document.getElementById("sidebar-toggle");
    const sidebar = document.querySelector(".sidebar");
    const overlay = document.getElementById("sidebar-overlay");

    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
        if (overlay) overlay.classList.toggle("active");
      });
    }

    if (overlay && sidebar) {
      overlay.addEventListener("click", () => {
        sidebar.classList.remove("open");
        overlay.classList.remove("active");
      });
    }

    // Auto-close on mobile/tablet when a nav item is clicked
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
      item.addEventListener("click", () => {
        if (window.innerWidth <= 992 && sidebar) {
          sidebar.classList.remove("open");
          if (overlay) overlay.classList.remove("active");
        }
      });
    });
  },

  setupTabNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
      item.addEventListener("click", () => {
        navItems.forEach(n => n.classList.remove("active"));
        item.classList.add("active");

        const targetTab = item.getAttribute("data-tab");
        document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
        const activePane = document.getElementById(`tab-${targetTab}`);
        if (activePane) {
          activePane.classList.add("active");

          // When a tab becomes visible, resize its Plotly charts so they fit perfectly
          setTimeout(() => {
            activePane.querySelectorAll(".chart-container").forEach(c => {
              if (window.Plotly && c) {
                try {
                  Plotly.Plots.resize(c);
                } catch (err) {}
              }
            });
            window.dispatchEvent(new Event("resize"));
          }, 60);
        }
      });
    });
  },

  async loadForecastViews() {
    const safeSet = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };

    try {
      const metrics = await API.getModelMetrics();
      if (metrics.solar_forecast) {
        safeSet("solar-mae", `${metrics.solar_forecast.mae_kw} kW`);
        safeSet("solar-rmse", `${metrics.solar_forecast.rmse_kw} kW`);
        safeSet("solar-r2", `${metrics.solar_forecast.r2_score}`);
      }
      if (metrics.load_forecast) {
        safeSet("load-mae", `${metrics.load_forecast.mae_kw} kW`);
        safeSet("load-rmse", `${metrics.load_forecast.rmse_kw} kW`);
        safeSet("load-mape", `${metrics.load_forecast.mape_pct}%`);
      }

      // Render default forecast charts
      const solarData = await API.getSolarForecast();
      const loadData = await API.getLoadForecast();

      if (solarData && solarData.timestamps && solarData.timestamps.length > 0) {
        Charts.renderForecast("chart-solar-forecast", solarData.timestamps, solarData.actual, solarData.predicted, "Solar Generation", "#f59e0b", "#ca8a04");
      }
      if (loadData && loadData.timestamps && loadData.timestamps.length > 0) {
        Charts.renderForecast("chart-load-forecast", loadData.timestamps, loadData.actual, loadData.predicted, "Household Demand", "#06b6d4", "#0284c7");
      }
    } catch (e) {
      console.warn("Forecast view loading warning:", e);
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  Dashboard.init();
});
