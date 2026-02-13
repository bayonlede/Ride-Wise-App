"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Legend,
} from "recharts";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Cloud,
  DollarSign,
  Loader2,
  TrendingUp,
  Users,
  Car,
  CloudRain,
  AlertTriangle,
} from "lucide-react";

const COLORS = ["#10b981", "#06b6d4", "#8b5cf6", "#f59e0b", "#ef4444", "#ec4899", "#6366f1", "#14b8a6"];

interface AnalyticsData {
  revenue_by_city: Array<{ name: string; value: number }>;
  trips_by_city: Array<{ name: string; value: number }>;
  top_10_drivers: Array<{ name: string; revenue: number }>;
  least_10_drivers: Array<{ name: string; revenue: number }>;
  revenue_by_period: Array<{ name: string; revenue: number }>;
  revenue_by_season: Array<{ name: string; revenue: number }>;
  revenue_by_age: Array<{ name: string; value: number }>;
  revenue_by_loyalty: Array<{ name: string; value: number }>;
  revenue_by_year: Array<{ name: string; revenue: number }>;
  trips_by_year: Array<{ name: string; trips: number }>;
  annual_revenue_vs_trips: Array<{ revenue: number; trips: number }>;
  revenue_by_vehicle: Array<{ name: string; revenue: number }>;
  trips_by_vehicle: Array<{ name: string; trips: number }>;
  surge_by_weather: Array<{ name: string; surge: number }>;
  fare_by_weather: Array<{ name: string; fare: number }>;
  driver_rating_churn: Array<{ rating: number; churn_prob: number }>;
  rider_rating_churn: Array<{ rating: number; churn_prob: number }>;
  ratings_by_city: Array<{ city: string; rating_by_driver: number; rating_by_rider: number }>;
  revenue_leakage: Array<{ name: string; lost_revenue: number }>;
  acceptance_by_weather: Array<{ name: string; acceptance_rate: number }>;
}

const emptyAnalytics: AnalyticsData = {
  revenue_by_city: [],
  trips_by_city: [],
  top_10_drivers: [],
  least_10_drivers: [],
  revenue_by_period: [],
  revenue_by_season: [],
  revenue_by_age: [],
  revenue_by_loyalty: [],
  revenue_by_year: [],
  trips_by_year: [],
  annual_revenue_vs_trips: [],
  revenue_by_vehicle: [],
  trips_by_vehicle: [],
  surge_by_weather: [],
  fare_by_weather: [],
  driver_rating_churn: [],
  rider_rating_churn: [],
  ratings_by_city: [],
  revenue_leakage: [],
  acceptance_by_weather: [],
};

function SectionCard({
  title,
  icon: Icon,
  children,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
}) {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500/20 to-brand-600/20 flex items-center justify-center text-brand-400">
          <Icon className="w-5 h-5" />
        </div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
      </div>
      {children}
    </div>
  );
}

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData>(emptyAnalytics);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [apiUrl, setApiUrl] = useState<string>("");

  useEffect(() => {
    const load = async () => {
      try {
        const configRes = await fetch("/api/config");
        const config = await configRes.json();
        const base = config?.apiUrl || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        setApiUrl(base);
        const res = await fetch(`${base}/analytics`);
        if (!res.ok) throw new Error("Analytics endpoint failed");
        const json = await res.json();
        setData(json);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load analytics");
        setData(emptyAnalytics);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <Loader2 className="w-10 h-10 text-brand-400 animate-spin" />
        <p className="text-slate-400">Loading analytics...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span className="text-sm font-medium">Dashboard</span>
              </Link>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center">
                  <BarChart3 className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">RideWise Analytics</h1>
                  <p className="text-xs text-slate-500">Exploratory Data Analysis</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 glass-card p-4 flex items-center gap-3 border-amber-500/30 bg-amber-500/5">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
            <p className="text-amber-200 text-sm">
              {error}. Ensure the API is running and trip data is available at <code className="text-amber-300">{apiUrl}/analytics</code>.
            </p>
          </div>
        )}

        {/* Revenue Overview */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <DollarSign className="w-6 h-6 text-brand-400" />
            Revenue Overview
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Total Revenue by City" icon={DollarSign}>
              {data.revenue_by_city.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={data.revenue_by_city}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {data.revenue_by_city.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Total Trips by City" icon={Activity}>
              {data.trips_by_city.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={data.trips_by_city}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {data.trips_by_city.map((_, i) => (
                        <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => [v.toLocaleString(), "Trips"]} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Drivers */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Users className="w-6 h-6 text-brand-400" />
            Driver Performance
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Top 10 Revenue Generating Drivers" icon={TrendingUp}>
              {data.top_10_drivers.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data.top_10_drivers} layout="vertical" margin={{ left: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#94a3b8" tickFormatter={(v) => `$${v}`} />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" width={70} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Bar dataKey="revenue" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Least 10 Revenue Generating Drivers" icon={Users}>
              {data.least_10_drivers.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={data.least_10_drivers} layout="vertical" margin={{ left: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#94a3b8" tickFormatter={(v) => `$${v}`} />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" width={70} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Bar dataKey="revenue" fill="#64748b" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Time: Period & Season */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Activity className="w-6 h-6 text-brand-400" />
            Revenue by Time
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Revenue by Period of Day" icon={BarChart3}>
              {data.revenue_by_period.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={data.revenue_by_period}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Line type="monotone" dataKey="revenue" stroke="#10b981" strokeWidth={2} dot={{ fill: "#10b981" }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Revenue by Season" icon={BarChart3}>
              {data.revenue_by_season.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={data.revenue_by_season}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Line type="monotone" dataKey="revenue" stroke="#06b6d4" strokeWidth={2} dot={{ fill: "#06b6d4" }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Demographics: Age & Loyalty */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Users className="w-6 h-6 text-brand-400" />
            Revenue by Rider Segment
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Revenue by Rider Age Group" icon={Users}>
              {data.revenue_by_age.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={data.revenue_by_age}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {data.revenue_by_age.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Revenue by Loyalty Status" icon={DollarSign}>
              {data.revenue_by_loyalty.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={data.revenue_by_loyalty}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {data.revenue_by_loyalty.map((_, i) => (
                        <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Yearly & Vehicle */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-brand-400" />
            Annual & Vehicle Trends
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <SectionCard title="Yearly Revenue" icon={DollarSign}>
              {data.revenue_by_year.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={data.revenue_by_year}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Bar dataKey="revenue" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Annual Trips" icon={Activity}>
              {data.trips_by_year.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={data.trips_by_year}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip formatter={(v: number) => [v.toLocaleString(), "Trips"]} />
                    <Bar dataKey="trips" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Revenue vs Trips (Annual)" icon={BarChart3}>
              {data.annual_revenue_vs_trips.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="revenue" name="Revenue" stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                    <YAxis dataKey="trips" name="Trips" stroke="#94a3b8" />
                    <Tooltip formatter={(v: number, n: string) => [n === "revenue" ? `$${v?.toLocaleString()}` : v?.toLocaleString(), n]} />
                    <Scatter data={data.annual_revenue_vs_trips} fill="#f59e0b" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <SectionCard title="Revenue by Vehicle Type" icon={Car}>
              {data.revenue_by_vehicle.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data.revenue_by_vehicle} layout="vertical" margin={{ left: 90 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" width={85} />
                    <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, "Revenue"]} />
                    <Bar dataKey="revenue" fill="#14b8a6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Trips by Vehicle Type" icon={Car}>
              {data.trips_by_vehicle.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data.trips_by_vehicle} layout="vertical" margin={{ left: 90 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" width={85} />
                    <Tooltip formatter={(v: number) => [v.toLocaleString(), "Trips"]} />
                    <Bar dataKey="trips" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Weather */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <CloudRain className="w-6 h-6 text-brand-400" />
            Weather Impact
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <SectionCard title="Avg Surge Multiplier by Weather" icon={Cloud}>
              {data.surge_by_weather.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={data.surge_by_weather}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip formatter={(v: number) => [v.toFixed(2), "Surge"]} />
                    <Bar dataKey="surge" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Avg Fare by Weather" icon={DollarSign}>
              {data.fare_by_weather.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={data.fare_by_weather}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v}`} />
                    <Tooltip formatter={(v: number) => [`$${v.toFixed(2)}`, "Fare"]} />
                    <Bar dataKey="fare" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Driver Acceptance Rate by Weather" icon={Cloud}>
              {data.acceptance_by_weather.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={data.acceptance_by_weather}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Acceptance"]} />
                    <Bar dataKey="acceptance_rate" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Ratings & Churn */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <Users className="w-6 h-6 text-brand-400" />
            Ratings & Churn
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <SectionCard title="Rider vs Driver Ratings by City" icon={BarChart3}>
              {data.ratings_by_city.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data.ratings_by_city} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="city" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" domain={[3, 5]} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="rating_by_driver" name="Driver rating" fill="#10b981" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="rating_by_rider" name="Rider rating" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Driver Rating vs Churn Probability" icon={TrendingUp}>
              {data.driver_rating_churn.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="rating" name="Driver rating" stroke="#94a3b8" domain={[1, 5]} />
                    <YAxis dataKey="churn_prob" name="Churn prob" stroke="#94a3b8" domain={[0, 1]} />
                    <Tooltip formatter={(v: number) => [typeof v === "number" ? v.toFixed(2) : v, "Value"]} />
                    <Scatter data={data.driver_rating_churn} fill="#10b981" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
            <SectionCard title="Rider Rating vs Churn Probability" icon={TrendingUp}>
              {data.rider_rating_churn.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="rating" name="Rider rating" stroke="#94a3b8" domain={[1, 5]} />
                    <YAxis dataKey="churn_prob" name="Churn prob" stroke="#94a3b8" domain={[0, 1]} />
                    <Tooltip formatter={(v: number) => [typeof v === "number" ? v.toFixed(2) : v, "Value"]} />
                    <Scatter data={data.rider_rating_churn} fill="#ec4899" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-slate-500 py-8 text-center">No data</p>
              )}
            </SectionCard>
          </div>
        </section>

        {/* Revenue Leakage */}
        <section className="mb-10">
          <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-amber-400" />
            Revenue Leakage (Low Acceptance)
          </h2>
          <SectionCard title="Estimated Lost Revenue by City" icon={DollarSign}>
            {data.revenue_leakage.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data.revenue_leakage}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" tickFormatter={(v) => `$${v / 1000}k`} />
                  <Tooltip formatter={(v: number) => [`$${v?.toLocaleString()}`, "Lost revenue"]} />
                  <Bar dataKey="lost_revenue" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-slate-500 py-8 text-center">No data</p>
            )}
          </SectionCard>
        </section>
      </main>

      <footer className="border-t border-slate-800/50 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
          <Link href="/" className="text-sm text-slate-500 hover:text-white transition-colors">
            ← Back to Churn Prediction
          </Link>
          <span className="text-sm text-slate-500">RideWise EDA from trip-level data</span>
        </div>
      </footer>
    </div>
  );
}
