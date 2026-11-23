/**
 * API Client for Smart Energy Management System Backend
 */

const API_BASE = window.location.origin;

const API = {
  async getDates() {
    const res = await fetch(`${API_BASE}/api/data/dates`);
    return await res.json();
  },

  async getCurrentStatus() {
    const res = await fetch(`${API_BASE}/api/system/current`);
    return await res.json();
  },

  async getSolarForecast(date) {
    const url = date ? `${API_BASE}/api/forecast/solar?date=${date}` : `${API_BASE}/api/forecast/solar`;
    const res = await fetch(url);
    return await res.json();
  },

  async getLoadForecast(date) {
    const url = date ? `${API_BASE}/api/forecast/load?date=${date}` : `${API_BASE}/api/forecast/load`;
    const res = await fetch(url);
    return await res.json();
  },

  async getModelMetrics() {
    const res = await fetch(`${API_BASE}/api/models/metrics`);
    return await res.json();
  },

  async runComparison(payload) {
    const res = await fetch(`${API_BASE}/api/run/comparison`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  },

  async getComparisonResults() {
    const res = await fetch(`${API_BASE}/api/results/comparison`);
    return await res.json();
  },

  async getComparisonTimeseries() {
    const res = await fetch(`${API_BASE}/api/results/timeseries`);
    return await res.json();
  },

  async getDecisionDetail(timestamp) {
    const res = await fetch(`${API_BASE}/api/results/decision?timestamp=${encodeURIComponent(timestamp)}`);
    return await res.json();
  },

  async getConfig() {
    const res = await fetch(`${API_BASE}/api/config`);
    return await res.json();
  },

  async saveConfig(payload) {
    const res = await fetch(`${API_BASE}/api/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  },

  async getScenarioSample(dayIdx, seed) {
    let url = `${API_BASE}/api/scenario/sample`;
    const params = [];
    if (dayIdx !== undefined && dayIdx !== null) params.push(`day_idx=${dayIdx}`);
    if (seed !== undefined && seed !== null) params.push(`seed=${seed}`);
    if (params.length > 0) url += `?${params.join("&")}`;
    const res = await fetch(url);
    return await res.json();
  },

  async getScenarioConfig() {
    const res = await fetch(`${API_BASE}/api/scenario/config`);
    return await res.json();
  },

  async getRlBenchmark() {
    const res = await fetch(`${API_BASE}/api/results/rl_benchmark`);
    return await res.json();
  }
};


