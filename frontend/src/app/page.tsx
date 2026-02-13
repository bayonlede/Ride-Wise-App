"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  RadialBarChart,
  RadialBar,
  Legend,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle,
  ChevronDown,
  Clock,
  DollarSign,
  Gauge,
  Heart,
  Info,
  Layers,
  Loader2,
  RefreshCw,
  Shield,
  Sparkles,
  Star,
  TrendingDown,
  TrendingUp,
  User,
  Users,
  Zap,
} from "lucide-react";

// Types
interface RiderFeatures {
  recency: number;
  frequency: number;
  monetary: number;
  surge_exposure: number;
  loyalty_status: number;
  churn_prob: number;
  rider_active_days: number;
  rating_by_rider: number;
}

interface PredictionResult {
  churn_probability: number;
  churn_classification: string;
  risk_level: string;
  confidence: number;
}

interface SHAPResult {
  feature_names: string[];
  shap_values: number[];
  base_value: number;
  prediction: number;
  feature_values: number[];
  top_positive_factors: Array<{ feature: string; shap_value: number; feature_value: number }>;
  top_negative_factors: Array<{ feature: string; shap_value: number; feature_value: number }>;
}

interface FeatureImportance {
  feature_names: string[];
  importance_values: number[];
  importance_type: string;
}

interface SampleRider {
  name: string;
  description: string;
  features: RiderFeatures;
}

// Fallback for build-time env (used if /api/config fails)
const FALLBACK_API_URL = process.env.NEXT_PUBLIC_API_URL || process.env.API_URL || "http://localhost:8000";

// Utility Functions
const cn = (...classes: (string | boolean | undefined)[]) => classes.filter(Boolean).join(" ");

const getRiskColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "low": return "#10b981";
    case "medium": return "#f59e0b";
    case "high": return "#f97316";
    case "critical": return "#ef4444";
    default: return "#64748b";
  }
};

const getLoyaltyLabel = (status: number) => {
  const labels = ["Bronze", "Silver", "Gold", "Platinum"];
  return labels[status] || "Unknown";
};

const formatFeatureName = (name: string) => {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

// Components
function Header({ apiUrl }: { apiUrl: string }) {
  const [apiStatus, setApiStatus] = useState<"loading" | "connected" | "error">("loading");

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiUrl}/health`);
        if (res.ok) setApiStatus("connected");
        else setApiStatus("error");
      } catch {
        setApiStatus("error");
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  return (
    <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">RideWise</h1>
              <p className="text-xs text-slate-500">Churn Prediction</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/analytics"
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-300 hover:text-white hover:border-slate-600 transition-colors text-sm font-medium"
            >
              <BarChart3 className="w-4 h-4" />
              Analytics
            </Link>
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50"
              title={apiStatus === "error" ? `Trying: ${apiUrl}` : undefined}
            >
              <div
                className={cn(
                  "w-2 h-2 rounded-full",
                  apiStatus === "connected" && "bg-brand-400 animate-pulse",
                  apiStatus === "loading" && "bg-amber-400 animate-pulse",
                  apiStatus === "error" && "bg-red-400"
                )}
              />
              <span className="text-xs text-slate-400">
                {apiStatus === "connected" && "API Connected"}
                {apiStatus === "loading" && "Connecting..."}
                {apiStatus === "error" && "API Offline"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  subValue,
  trend,
  color = "brand",
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  subValue?: string;
  trend?: "up" | "down";
  color?: "brand" | "amber" | "red" | "blue";
}) {
  const colorClasses = {
    brand: "from-brand-500/20 to-brand-600/20 text-brand-400",
    amber: "from-amber-500/20 to-amber-600/20 text-amber-400",
    red: "from-red-500/20 to-red-600/20 text-red-400",
    blue: "from-blue-500/20 to-blue-600/20 text-blue-400",
  };

  return (
    <div className="glass-card p-5 group hover:border-slate-700/50 transition-all duration-300">
      <div className="flex items-start justify-between">
        <div
          className={cn(
            "w-10 h-10 rounded-xl bg-gradient-to-br flex items-center justify-center",
            colorClasses[color]
          )}
        >
          <Icon className="w-5 h-5" />
        </div>
        {trend && (
          <div
            className={cn(
              "flex items-center gap-1 text-xs font-medium",
              trend === "up" ? "text-brand-400" : "text-red-400"
            )}
          >
            {trend === "up" ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className="text-2xl font-bold text-white">{value}</p>
        <p className="text-sm text-slate-500 mt-0.5">{label}</p>
        {subValue && <p className="text-xs text-slate-600 mt-1">{subValue}</p>}
      </div>
    </div>
  );
}

function PredictionForm({
  features,
  setFeatures,
  onPredict,
  loading,
  sampleRiders,
}: {
  features: RiderFeatures;
  setFeatures: (f: RiderFeatures) => void;
  onPredict: () => void;
  loading: boolean;
  sampleRiders: SampleRider[];
}) {
  const [selectedSample, setSelectedSample] = useState<string>("");

  const handleSampleSelect = (sampleName: string) => {
    setSelectedSample(sampleName);
    const sample = sampleRiders.find((s) => s.name === sampleName);
    if (sample) {
      setFeatures(sample.features);
    }
  };

  const handleInputChange = (field: keyof RiderFeatures, value: number) => {
    setFeatures({ ...features, [field]: value });
    setSelectedSample("");
  };

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <User className="w-5 h-5 text-brand-400" />
            Rider Profile
          </h2>
          <p className="text-sm text-slate-500 mt-1">Enter rider metrics for prediction</p>
        </div>
      </div>

      {/* Sample Selector */}
      {sampleRiders.length > 0 && (
        <div className="mb-6">
          <label className="label">Quick Load Sample</label>
          <div className="relative">
            <select
              value={selectedSample}
              onChange={(e) => handleSampleSelect(e.target.value)}
              className="input-field appearance-none pr-10 cursor-pointer"
            >
              <option value="">Select a sample rider...</option>
              {sampleRiders.map((sample) => (
                <option key={sample.name} value={sample.name}>
                  {sample.name}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500 pointer-events-none" />
          </div>
          {selectedSample && (
            <p className="text-xs text-slate-500 mt-2">
              {sampleRiders.find((s) => s.name === selectedSample)?.description}
            </p>
          )}
        </div>
      )}

      {/* Form Grid */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label flex items-center gap-2">
            <Clock className="w-3.5 h-3.5" />
            Recency (days)
          </label>
          <input
            type="number"
            min="0"
            value={features.recency}
            onChange={(e) => handleInputChange("recency", Number(e.target.value))}
            className="input-field"
            placeholder="Days since last ride"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" />
            Frequency (trips)
          </label>
          <input
            type="number"
            min="0"
            value={features.frequency}
            onChange={(e) => handleInputChange("frequency", Number(e.target.value))}
            className="input-field"
            placeholder="Total trips"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <DollarSign className="w-3.5 h-3.5" />
            Monetary (total spent)
          </label>
          <input
            type="number"
            min="0"
            step="0.01"
            value={features.monetary}
            onChange={(e) => handleInputChange("monetary", Number(e.target.value))}
            className="input-field"
            placeholder="Total spending"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <Zap className="w-3.5 h-3.5" />
            Surge Exposure (0-1)
          </label>
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={features.surge_exposure}
            onChange={(e) => handleInputChange("surge_exposure", Number(e.target.value))}
            className="input-field"
            placeholder="% rides during surge"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <Shield className="w-3.5 h-3.5" />
            Loyalty Status
          </label>
          <select
            value={features.loyalty_status}
            onChange={(e) => handleInputChange("loyalty_status", Number(e.target.value))}
            className="input-field appearance-none cursor-pointer"
          >
            <option value={0}>Bronze</option>
            <option value={1}>Silver</option>
            <option value={2}>Gold</option>
            <option value={3}>Platinum</option>
          </select>
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5" />
            Historical Churn Score
          </label>
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={features.churn_prob}
            onChange={(e) => handleInputChange("churn_prob", Number(e.target.value))}
            className="input-field"
            placeholder="0-1 churn probability"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <Users className="w-3.5 h-3.5" />
            Active Days
          </label>
          <input
            type="number"
            min="1"
            value={features.rider_active_days}
            onChange={(e) => handleInputChange("rider_active_days", Number(e.target.value))}
            className="input-field"
            placeholder="Days since signup"
          />
        </div>

        <div>
          <label className="label flex items-center gap-2">
            <Star className="w-3.5 h-3.5" />
            Rating by Rider (1-5)
          </label>
          <input
            type="number"
            min="1"
            max="5"
            step="0.1"
            value={features.rating_by_rider}
            onChange={(e) => handleInputChange("rating_by_rider", Number(e.target.value))}
            className="input-field"
            placeholder="Average rating given"
          />
        </div>
      </div>

      {/* Predict Button */}
      <button
        onClick={onPredict}
        disabled={loading}
        className="btn-primary w-full mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Brain className="w-4 h-4" />
            Predict Churn Risk
          </>
        )}
      </button>
    </div>
  );
}

function PredictionResult({ prediction }: { prediction: PredictionResult }) {
  const riskConfig = {
    Low: { color: "#10b981", bg: "risk-badge-low", icon: CheckCircle },
    Medium: { color: "#f59e0b", bg: "risk-badge-medium", icon: Info },
    High: { color: "#f97316", bg: "risk-badge-high", icon: AlertTriangle },
    Critical: { color: "#ef4444", bg: "risk-badge-critical", icon: AlertTriangle },
  };

  const config = riskConfig[prediction.risk_level as keyof typeof riskConfig] || riskConfig.Medium;
  const Icon = config.icon;
  const probability = Math.round(prediction.churn_probability * 100);

  const gaugeData = [
    {
      name: "Churn",
      value: probability,
      fill: config.color,
    },
  ];

  return (
    <div className="glass-card p-6 gradient-border">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Gauge className="w-5 h-5 text-brand-400" />
          Prediction Result
        </h2>
        <span
          className={cn(
            "px-3 py-1 text-xs font-semibold rounded-full border",
            config.bg
          )}
        >
          {prediction.risk_level} Risk
        </span>
      </div>

      {/* Gauge Chart */}
      <div className="flex justify-center mb-6">
        <div className="relative w-48 h-40">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              cx="50%"
              cy="100%"
              innerRadius="40%"
              outerRadius="100%"
              barSize={18}
              data={gaugeData}
              startAngle={180}
              endAngle={0}
            >
              <RadialBar
                background={{ fill: "#020617" }}
                dataKey="value"
                cornerRadius={10}
              />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center pb-4">
            <span className="text-4xl font-bold leading-none" style={{ color: config.color }}>
              {probability}%
            </span>
            <span className="mt-1 text-sm text-slate-500">Churn Risk</span>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-3 bg-slate-800/30 rounded-xl">
          <p className="text-2xl font-bold text-white">{prediction.churn_classification}</p>
          <p className="text-xs text-slate-500 mt-1">Classification</p>
        </div>
        <div className="text-center p-3 bg-slate-800/30 rounded-xl">
          <p className="text-2xl font-bold" style={{ color: config.color }}>
            <Icon className="w-6 h-6 mx-auto" />
          </p>
          <p className="text-xs text-slate-500 mt-1">{prediction.risk_level}</p>
        </div>
        <div className="text-center p-3 bg-slate-800/30 rounded-xl">
          <p className="text-2xl font-bold text-blue-400">
            {Math.round(prediction.confidence * 100)}%
          </p>
          <p className="text-xs text-slate-500 mt-1">Confidence</p>
        </div>
      </div>
    </div>
  );
}

function SHAPExplanation({ shap }: { shap: SHAPResult }) {
  const chartData = shap.feature_names
    .map((name, idx) => ({
      name: formatFeatureName(name),
      value: shap.shap_values[idx],
      fill: shap.shap_values[idx] > 0 ? "#ef4444" : "#10b981",
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brand-400" />
          SHAP Explanation
        </h2>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-500"></span>
            Increases Churn
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-brand-500"></span>
            Decreases Churn
          </span>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" stroke="#64748b" fontSize={12} />
            <YAxis
              dataKey="name"
              type="category"
              stroke="#64748b"
              fontSize={11}
              width={90}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
              }}
              labelStyle={{ color: "#f1f5f9" }}
              formatter={(value: number) => [value.toFixed(4), "SHAP Value"]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Key Factors */}
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm font-medium text-red-400 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Increasing Churn Risk
          </h3>
          <div className="space-y-2">
            {shap.top_positive_factors.slice(0, 3).map((factor) => (
              <div
                key={factor.feature}
                className="flex items-center justify-between p-3 bg-red-500/10 border border-red-500/20 rounded-lg"
              >
                <span className="text-sm text-slate-300">
                  {formatFeatureName(factor.feature)}
                </span>
                <span className="text-sm font-mono text-red-400">
                  +{factor.shap_value.toFixed(4)}
                </span>
              </div>
            ))}
            {shap.top_positive_factors.length === 0 && (
              <p className="text-sm text-slate-500 italic">No positive factors</p>
            )}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium text-brand-400 mb-3 flex items-center gap-2">
            <TrendingDown className="w-4 h-4" />
            Decreasing Churn Risk
          </h3>
          <div className="space-y-2">
            {shap.top_negative_factors.slice(0, 3).map((factor) => (
              <div
                key={factor.feature}
                className="flex items-center justify-between p-3 bg-brand-500/10 border border-brand-500/20 rounded-lg"
              >
                <span className="text-sm text-slate-300">
                  {formatFeatureName(factor.feature)}
                </span>
                <span className="text-sm font-mono text-brand-400">
                  {factor.shap_value.toFixed(4)}
                </span>
              </div>
            ))}
            {shap.top_negative_factors.length === 0 && (
              <p className="text-sm text-slate-500 italic">No negative factors</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function FeatureImportanceChart({ importance }: { importance: FeatureImportance }) {
  const chartData = importance.feature_names
    .map((name, idx) => ({
      name: formatFeatureName(name),
      value: importance.importance_values[idx],
    }))
    .sort((a, b) => b.value - a.value);

  const colors = [
    "#10b981", "#06b6d4", "#3b82f6", "#8b5cf6",
    "#ec4899", "#f59e0b", "#ef4444", "#84cc16", "#14b8a6",
  ];

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-brand-400" />
          Global Feature Importance
        </h2>
      </div>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis
              type="number"
              stroke="#64748b"
              fontSize={12}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <YAxis
              dataKey="name"
              type="category"
              stroke="#64748b"
              fontSize={11}
              width={90}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
              }}
              labelStyle={{ color: "#f1f5f9" }}
              formatter={(value: number) => [value.toFixed(4), "Importance"]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-slate-500 mt-4 text-center">
        {importance.importance_type}
      </p>
    </div>
  );
}

function Recommendations({ riskLevel }: { riskLevel: string }) {
  const recommendations = {
    Low: {
      title: "Healthy Engagement",
      icon: Heart,
      color: "brand",
      items: [
        "Send personalized thank you messages for loyalty",
        "Offer referral program incentives",
        "Invite to exclusive loyalty events",
        "Provide early access to new features",
      ],
    },
    Medium: {
      title: "Monitor & Engage",
      icon: Activity,
      color: "amber",
      items: [
        "Send regular engagement communications",
        "Offer occasional promotional discounts",
        "Monitor activity trends closely",
        "Request feedback to understand concerns",
      ],
    },
    High: {
      title: "Proactive Outreach",
      icon: AlertTriangle,
      color: "orange",
      items: [
        "Send personalized win-back campaign",
        "Offer significant discount or upgrade",
        "Proactive customer service outreach",
        "Identify and address pain points",
      ],
    },
    Critical: {
      title: "Immediate Action Required",
      icon: AlertTriangle,
      color: "red",
      items: [
        "Personal call from customer success team",
        "Offer exclusive retention package",
        "Conduct exit interview if leaving",
        "Implement emergency retention protocol",
      ],
    },
  };

  const rec = recommendations[riskLevel as keyof typeof recommendations] || recommendations.Medium;
  const Icon = rec.icon;

  const colorClasses = {
    brand: "from-brand-500/20 to-brand-600/20 border-brand-500/30 text-brand-400",
    amber: "from-amber-500/20 to-amber-600/20 border-amber-500/30 text-amber-400",
    orange: "from-orange-500/20 to-orange-600/20 border-orange-500/30 text-orange-400",
    red: "from-red-500/20 to-red-600/20 border-red-500/30 text-red-400",
  };

  return (
    <div
      className={cn(
        "glass-card p-6 bg-gradient-to-br border",
        colorClasses[rec.color as keyof typeof colorClasses]
      )}
    >
      <div className="flex items-center gap-3 mb-4">
        <Icon className="w-6 h-6" />
        <h2 className="text-lg font-semibold text-white">{rec.title}</h2>
      </div>

      <ul className="space-y-3">
        {rec.items.map((item, idx) => (
          <li key={idx} className="flex items-start gap-3">
            <ArrowRight className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span className="text-sm text-slate-300">{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Main Component
export default function Dashboard() {
  const [features, setFeatures] = useState<RiderFeatures>({
    recency: 15,
    frequency: 20,
    monetary: 250,
    surge_exposure: 0.25,
    loyalty_status: 1,
    churn_prob: 0.3,
    rider_active_days: 200,
    rating_by_rider: 4.2,
  });

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [shapResult, setShapResult] = useState<SHAPResult | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
  const [sampleRiders, setSampleRiders] = useState<SampleRider[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [apiUrl, setApiUrl] = useState<string>(FALLBACK_API_URL);

  // Fetch API URL from server at runtime (Railway injects vars at runtime)
  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((data) => setApiUrl(data.apiUrl || FALLBACK_API_URL))
      .catch(() => {});
  }, []);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Load sample riders
        const samplesRes = await fetch(`${apiUrl}/sample-riders`);
        if (samplesRes.ok) {
          const data = await samplesRes.json();
          setSampleRiders(data.samples);
        }

        // Load feature importance
        const importanceRes = await fetch(`${apiUrl}/feature-importance`);
        if (importanceRes.ok) {
          const data = await importanceRes.json();
          setFeatureImportance(data);
        }
      } catch (e) {
        console.error("Failed to load initial data:", e);
      }
    };

    loadData();
  }, [apiUrl]);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Get prediction
      const predRes = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      if (!predRes.ok) throw new Error("Prediction failed");
      const predData = await predRes.json();
      setPrediction(predData);

      // Get SHAP explanation
      const shapRes = await fetch(`${apiUrl}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      if (shapRes.ok) {
        const shapData = await shapRes.json();
        setShapResult(shapData);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "An error occurred";
      const isFetch = msg.toLowerCase().includes("fetch") || msg.toLowerCase().includes("network");
      setError(
        isFetch
          ? `Failed to fetch: Could not reach API. Add NEXT_PUBLIC_API_URL (or API_URL) in Railway Variables for this service.`
          : msg
      );
    } finally {
      setLoading(false);
    }
  }, [features, apiUrl]);

  return (
    <div className="min-h-screen">
      <Header apiUrl={apiUrl} />

      {/* Hero Section */}
      <section className="relative py-12 overflow-hidden">
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-brand-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-brand-500/10 border border-brand-500/20 rounded-full text-brand-400 text-sm font-medium mb-6">
              <Brain className="w-4 h-4" />
              AI-Powered Analytics
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Churn Prediction
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-cyan-400">
                Dashboard
              </span>
            </h1>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Predict rider churn using machine learning, understand risk factors with SHAP explanations,
              and take proactive retention actions.
            </p>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
            <StatCard icon={Users} label="Total Riders" value="10,000" subValue="In dataset" color="brand" />
            <StatCard icon={Activity} label="Trips Analyzed" value="200K" subValue="Historical data" color="blue" />
            <StatCard icon={Shield} label="Model Accuracy" value="99.9%" subValue="Test set" color="brand" />
            <StatCard icon={Layers} label="Features" value="9" subValue="Input variables" color="amber" />
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Column - Input */}
          <div className="space-y-6">
            <PredictionForm
              features={features}
              setFeatures={setFeatures}
              onPredict={handlePredict}
              loading={loading}
              sampleRiders={sampleRiders}
            />

            {featureImportance && <FeatureImportanceChart importance={featureImportance} />}
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {prediction ? (
              <>
                <PredictionResult prediction={prediction} />
                {shapResult && <SHAPExplanation shap={shapResult} />}
                <Recommendations riskLevel={prediction.risk_level} />
              </>
            ) : (
              <div className="glass-card p-12 text-center">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-slate-800/50 flex items-center justify-center">
                  <Brain className="w-10 h-10 text-slate-600" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
                <p className="text-slate-500 max-w-sm mx-auto">
                  Enter rider metrics on the left or select a sample profile, then click
                  &quot;Predict Churn Risk&quot; to see results.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center">
                <Activity className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-white">RideWise Analytics</span>
            </div>
            <p className="text-sm text-slate-500">
              Powered by Machine Learning &bull; RandomForest + SHAP
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
