"use client";

import { useState, useEffect } from "react";

const navLinks = [
  { href: "#home", label: "Home" },
  { href: "#features", label: "Features" },
  { href: "#about", label: "About" },
  { href: "#contact", label: "Contact" },
];

const features = [
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
    title: "Churn Prediction",
    description:
      "Identify at-risk customers before they leave with our ML-powered prediction engine. Achieve up to 92% accuracy in forecasting user churn.",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
        />
      </svg>
    ),
    title: "Smart Segmentation",
    description:
      "Automatically group customers into meaningful segments—commuters, weekend riders, occasional users—for targeted campaigns.",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M13 10V3L4 14h7v7l9-11h-7z"
        />
      </svg>
    ),
    title: "Real-Time Insights",
    description:
      "Get instant risk scoring and actionable insights through our API-driven dashboards. Make data-driven decisions in seconds.",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
    title: "Revenue Optimization",
    description:
      "Maximize customer lifetime value with personalized retention strategies. Reduce churn costs by up to 40%.",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        />
      </svg>
    ),
    title: "RFM Analysis",
    description:
      "Leverage Recency, Frequency, and Monetary analysis to understand customer behavior and prioritize high-value users.",
  },
  {
    icon: (
      <svg
        className="w-6 h-6"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
        />
      </svg>
    ),
    title: "Interactive Dashboards",
    description:
      "Visualize trends, monitor KPIs, and track campaign performance with beautiful, customizable Streamlit dashboards.",
  },
];

const steps = [
  {
    number: "01",
    title: "Connect Your Data",
    description:
      "Integrate your customer, trip, and session data through our secure API or batch upload.",
  },
  {
    number: "02",
    title: "Analyze & Segment",
    description:
      "Our ML models process your data to identify patterns, predict churn, and segment customers.",
  },
  {
    number: "03",
    title: "Take Action",
    description:
      "Receive actionable insights and automated recommendations to retain at-risk customers.",
  },
  {
    number: "04",
    title: "Measure Impact",
    description:
      "Track retention improvements and ROI through comprehensive analytics dashboards.",
  },
];

export default function Home() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (
    e: React.MouseEvent<HTMLAnchorElement>,
    href: string
  ) => {
    e.preventDefault();
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
    setIsMenuOpen(false);
  };

  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled
            ? "bg-dark-950/90 backdrop-blur-lg border-b border-dark-800"
            : "bg-transparent"
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 md:h-20">
            {/* Logo */}
            <a
              href="#home"
              onClick={(e) => scrollToSection(e, "#home")}
              className="flex items-center space-x-2"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-emerald-400 rounded-xl flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-dark-950"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                  />
                </svg>
              </div>
              <span className="text-xl font-bold tracking-tight">RideWise</span>
            </a>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center space-x-8">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={(e) => scrollToSection(e, link.href)}
                  className="text-sm font-medium text-dark-300 hover:text-primary-400 transition-colors"
                >
                  {link.label}
                </a>
              ))}
              <a href="#contact" className="btn-primary text-sm">
                Get Started
              </a>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 text-dark-300 hover:text-white"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div
          className={`md:hidden transition-all duration-300 overflow-hidden ${
            isMenuOpen ? "max-h-96" : "max-h-0"
          }`}
        >
          <div className="bg-dark-900/95 backdrop-blur-lg border-t border-dark-800 px-4 py-4 space-y-3">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={(e) => scrollToSection(e, link.href)}
                className="block py-2 text-base font-medium text-dark-300 hover:text-primary-400 transition-colors"
              >
                {link.label}
              </a>
            ))}
            <a
              href="#contact"
              onClick={(e) => scrollToSection(e, "#contact")}
              className="btn-primary w-full text-center mt-4"
            >
              Get Started
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section
        id="home"
        className="relative min-h-screen flex items-center pt-20 overflow-hidden"
      >
        {/* Background Effects */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-dark-900/50 via-dark-950 to-dark-950" />
          {/* Grid Pattern */}
          <div
            className="absolute inset-0 opacity-[0.02]"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
            }}
          />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            <div className="space-y-8">
              <div className="inline-flex items-center px-4 py-2 bg-primary-500/10 border border-primary-500/20 rounded-full text-primary-400 text-sm font-medium animate-fade-in">
                <span className="w-2 h-2 bg-primary-400 rounded-full mr-2 animate-pulse" />
                AI-Powered Customer Analytics
              </div>

              <h1 className="section-heading text-4xl md:text-5xl lg:text-6xl animate-slide-up">
                Stop Losing Customers.
                <br />
                <span className="gradient-text">Start Predicting Churn.</span>
              </h1>

              <p className="section-subheading animate-slide-up stagger-2">
                RideWise empowers mobility companies to reduce customer churn by
                up to 40% through intelligent analytics, ML-driven predictions,
                and actionable retention strategies.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 animate-slide-up stagger-3">
                <a
                  href="#contact"
                  onClick={(e) => scrollToSection(e, "#contact")}
                  className="btn-primary"
                >
                  Start Free Trial
                  <svg
                    className="w-4 h-4 ml-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M17 8l4 4m0 0l-4 4m4-4H3"
                    />
                  </svg>
                </a>
                <a
                  href="#about"
                  onClick={(e) => scrollToSection(e, "#about")}
                  className="btn-secondary"
                >
                  See How It Works
                </a>
              </div>

              <div className="flex items-center gap-8 pt-4 animate-slide-up stagger-4">
                <div>
                  <div className="text-3xl font-bold text-white">200K+</div>
                  <div className="text-sm text-dark-400">Active Customers</div>
                </div>
                <div className="w-px h-12 bg-dark-700" />
                <div>
                  <div className="text-3xl font-bold text-white">800K</div>
                  <div className="text-sm text-dark-400">Monthly Trips</div>
                </div>
                <div className="w-px h-12 bg-dark-700" />
                <div>
                  <div className="text-3xl font-bold text-white">92%</div>
                  <div className="text-sm text-dark-400">Prediction Accuracy</div>
                </div>
              </div>
            </div>

            {/* Hero Visual */}
            <div className="relative hidden lg:block animate-slide-in-right">
              <div className="relative w-full aspect-square max-w-lg mx-auto">
                {/* Main Dashboard Card */}
                <div className="absolute inset-0 bg-gradient-to-br from-dark-800 to-dark-900 rounded-3xl border border-dark-700 shadow-2xl p-6 animate-float">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500" />
                      <div className="w-3 h-3 rounded-full bg-yellow-500" />
                      <div className="w-3 h-3 rounded-full bg-green-500" />
                    </div>
                    <span className="text-xs text-dark-500">
                      Customer Dashboard
                    </span>
                  </div>

                  {/* Mock Chart */}
                  <div className="space-y-4">
                    <div className="flex justify-between items-end h-32 gap-2">
                      {[40, 65, 45, 80, 55, 70, 90].map((height, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-gradient-to-t from-primary-500/20 to-primary-500/60 rounded-t"
                          style={{ height: `${height}%` }}
                        />
                      ))}
                    </div>
                    <div className="flex justify-between text-xs text-dark-500">
                      <span>Mon</span>
                      <span>Tue</span>
                      <span>Wed</span>
                      <span>Thu</span>
                      <span>Fri</span>
                      <span>Sat</span>
                      <span>Sun</span>
                    </div>
                  </div>

                  {/* Stats Row */}
                  <div className="grid grid-cols-2 gap-4 mt-6">
                    <div className="bg-dark-800/50 rounded-xl p-4">
                      <div className="text-xs text-dark-400 mb-1">
                        Churn Risk
                      </div>
                      <div className="text-xl font-bold text-red-400">
                        -12.4%
                      </div>
                    </div>
                    <div className="bg-dark-800/50 rounded-xl p-4">
                      <div className="text-xs text-dark-400 mb-1">Retention</div>
                      <div className="text-xl font-bold text-primary-400">
                        +8.7%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Floating Badge */}
                <div className="absolute -bottom-4 -left-4 bg-dark-800 border border-dark-700 rounded-2xl p-4 shadow-xl animate-float delay-500">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-primary-500/20 rounded-full flex items-center justify-center">
                      <svg
                        className="w-5 h-5 text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                        />
                      </svg>
                    </div>
                    <div>
                      <div className="text-sm font-medium">Revenue Up</div>
                      <div className="text-xs text-dark-400">
                        +24% this quarter
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 md:py-32 relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-dark-950 via-dark-900/50 to-dark-950" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-flex items-center px-4 py-2 bg-primary-500/10 border border-primary-500/20 rounded-full text-primary-400 text-sm font-medium mb-6">
              Powerful Features
            </div>
            <h2 className="section-heading mb-4">
              Everything You Need to
              <br />
              <span className="gradient-text">Retain Customers</span>
            </h2>
            <p className="section-subheading mx-auto">
              Our comprehensive platform combines machine learning, real-time
              analytics, and actionable insights to help you understand and
              retain your customers.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="card group hover:border-primary-500/30"
              >
                <div className="w-12 h-12 bg-primary-500/10 rounded-xl flex items-center justify-center text-primary-400 mb-4 group-hover:bg-primary-500/20 transition-colors">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-dark-400 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About / How It Works Section */}
      <section id="about" className="py-20 md:py-32 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <div className="inline-flex items-center px-4 py-2 bg-primary-500/10 border border-primary-500/20 rounded-full text-primary-400 text-sm font-medium mb-6">
                How It Works
              </div>
              <h2 className="section-heading mb-6">
                From Data to
                <br />
                <span className="gradient-text">Actionable Insights</span>
              </h2>
              <p className="section-subheading mb-8">
                RideWise transforms your raw customer data into powerful
                predictions and personalized retention strategies. Our platform
                processes over 200,000 customer records and 800,000 monthly
                trips to deliver insights that matter.
              </p>

              <div className="space-y-6">
                {steps.map((step, index) => (
                  <div key={index} className="flex gap-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-primary-500 to-emerald-500 rounded-xl flex items-center justify-center text-dark-950 font-bold text-sm">
                      {step.number}
                    </div>
                    <div>
                      <h3 className="font-semibold mb-1">{step.title}</h3>
                      <p className="text-sm text-dark-400">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Visual Element */}
            <div className="relative hidden lg:block">
              <div className="bg-gradient-to-br from-dark-800 to-dark-900 rounded-3xl border border-dark-700 p-8">
                <h3 className="text-lg font-semibold mb-6">
                  Customer Segmentation
                </h3>

                {/* Segments */}
                <div className="space-y-4">
                  {[
                    {
                      label: "Daily Commuters",
                      value: 42,
                      color: "bg-primary-500",
                    },
                    {
                      label: "Weekend Riders",
                      value: 28,
                      color: "bg-emerald-500",
                    },
                    {
                      label: "Occasional Users",
                      value: 18,
                      color: "bg-blue-500",
                    },
                    { label: "At Risk", value: 12, color: "bg-red-500" },
                  ].map((segment, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-dark-300">{segment.label}</span>
                        <span className="text-dark-400">{segment.value}%</span>
                      </div>
                      <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${segment.color} rounded-full transition-all duration-1000`}
                          style={{ width: `${segment.value}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Cities */}
                <div className="mt-8 pt-6 border-t border-dark-700">
                  <h4 className="text-sm font-medium text-dark-400 mb-4">
                    Operating Cities
                  </h4>
                  <div className="flex gap-3">
                    {["Cairo", "Nairobi", "Lagos"].map((city, i) => (
                      <span
                        key={i}
                        className="px-3 py-1.5 bg-dark-700/50 rounded-lg text-sm text-dark-300"
                      >
                        {city}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 md:py-32 relative overflow-hidden">
        <div className="absolute inset-0 -z-10">
          <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 via-transparent to-emerald-500/10" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary-500/5 rounded-full blur-3xl" />
        </div>

        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="section-heading mb-6">
            Ready to Transform Your
            <br />
            <span className="gradient-text">Customer Retention?</span>
          </h2>
          <p className="section-subheading mx-auto mb-10">
            Join leading mobility companies using RideWise to predict churn,
            segment customers, and boost lifetime value. Start your free trial
            today.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="#contact"
              onClick={(e) => scrollToSection(e, "#contact")}
              className="btn-primary text-lg px-8 py-4"
            >
              Start Free Trial
              <svg
                className="w-5 h-5 ml-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 8l4 4m0 0l-4 4m4-4H3"
                />
              </svg>
            </a>
            <a href="#features" className="btn-secondary text-lg px-8 py-4">
              Explore Features
            </a>
          </div>

          <p className="mt-8 text-sm text-dark-500">
            No credit card required • 14-day free trial • Cancel anytime
          </p>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20 md:py-32 relative">
        <div className="absolute inset-0 -z-10 bg-gradient-to-b from-dark-950 via-dark-900/30 to-dark-950" />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16">
            <div>
              <div className="inline-flex items-center px-4 py-2 bg-primary-500/10 border border-primary-500/20 rounded-full text-primary-400 text-sm font-medium mb-6">
                Get In Touch
              </div>
              <h2 className="section-heading mb-6">
                Let&apos;s Start Your
                <br />
                <span className="gradient-text">Analytics Journey</span>
              </h2>
              <p className="section-subheading mb-8">
                Have questions about RideWise? Want to see a demo? Our team is
                here to help you reduce churn and maximize customer lifetime
                value.
              </p>

              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-primary-500/10 rounded-xl flex items-center justify-center text-primary-400">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                      />
                    </svg>
                  </div>
                  <div>
                    <div className="text-sm text-dark-400">Email</div>
                    <div className="font-medium">hello@ridewise.io</div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-primary-500/10 rounded-xl flex items-center justify-center text-primary-400">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                  </div>
                  <div>
                    <div className="text-sm text-dark-400">Locations</div>
                    <div className="font-medium">Cairo • Nairobi • Lagos</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Contact Form */}
            <div className="bg-dark-900/50 backdrop-blur-sm border border-dark-800 rounded-2xl p-8">
              <form className="space-y-6">
                <div className="grid sm:grid-cols-2 gap-6">
                  <div>
                    <label
                      htmlFor="name"
                      className="block text-sm font-medium text-dark-300 mb-2"
                    >
                      Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 transition-colors"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="email"
                      className="block text-sm font-medium text-dark-300 mb-2"
                    >
                      Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 transition-colors"
                      placeholder="you@company.com"
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="company"
                    className="block text-sm font-medium text-dark-300 mb-2"
                  >
                    Company
                  </label>
                  <input
                    type="text"
                    id="company"
                    className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 transition-colors"
                    placeholder="Your company"
                  />
                </div>

                <div>
                  <label
                    htmlFor="message"
                    className="block text-sm font-medium text-dark-300 mb-2"
                  >
                    Message
                  </label>
                  <textarea
                    id="message"
                    rows={4}
                    className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl text-white placeholder-dark-500 focus:outline-none focus:border-primary-500 transition-colors resize-none"
                    placeholder="Tell us about your needs..."
                  />
                </div>

                <button type="submit" className="btn-primary w-full">
                  Send Message
                  <svg
                    className="w-4 h-4 ml-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M14 5l7 7m0 0l-7 7m7-7H3"
                    />
                  </svg>
                </button>
              </form>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            {/* Logo */}
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-400 to-emerald-400 rounded-lg flex items-center justify-center">
                <svg
                  className="w-4 h-4 text-dark-950"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                  />
                </svg>
              </div>
              <span className="font-bold">RideWise</span>
            </div>

            {/* Footer Links */}
            <div className="flex items-center gap-6 text-sm">
              <a
                href="#"
                className="text-dark-400 hover:text-primary-400 transition-colors"
              >
                Privacy
              </a>
              <a
                href="#"
                className="text-dark-400 hover:text-primary-400 transition-colors"
              >
                Terms
              </a>
              <a
                href="#contact"
                onClick={(e) => scrollToSection(e, "#contact")}
                className="text-dark-400 hover:text-primary-400 transition-colors"
              >
                Contact
              </a>
            </div>

            {/* Copyright */}
            <div className="text-sm text-dark-500">
              © {new Date().getFullYear()} RideWise. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
