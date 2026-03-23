import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import Head from "next/head";
import Link from "next/link";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.6 },
  }),
};

const PaperCard = ({ number, title, authors, abstract, status, tags, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex items-start justify-between mb-3">
      <span className="text-primary font-mono text-xs">Paper {number}</span>
      <span
        className={`text-xs font-mono px-2 py-0.5 rounded ${
          status === "Preprint"
            ? "bg-accent/20 text-accentLight"
            : "bg-emerald/20 text-emerald"
        }`}
      >
        {status}
      </span>
    </div>
    <h3 className="text-light font-bold text-lg mb-2 leading-snug">{title}</h3>
    <p className="text-lightMuted text-xs mb-3">{authors}</p>
    <p className="text-lightMuted text-sm leading-relaxed mb-4">{abstract}</p>
    <div className="flex flex-wrap gap-2">
      {tags.map((tag, i) => (
        <span
          key={i}
          className="text-xs font-mono px-2 py-0.5 rounded bg-darkAlt text-lightMuted border border-primary/10"
        >
          {tag}
        </span>
      ))}
    </div>
  </motion.div>
);

const ValidationPanel = ({ title, panels, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <h3 className="text-light font-semibold text-sm mb-3">{title}</h3>
    <div className="space-y-2">
      {panels.map((panel, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-primary/60 flex-shrink-0" />
          <span className="text-lightMuted text-sm">{panel}</span>
        </div>
      ))}
    </div>
  </motion.div>
);

const DatasetCard = ({ name, source, records, fields, delay }) => (
  <motion.div
    className="glass-card glow-border"
    custom={delay}
    variants={fadeUp}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true }}
  >
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-light font-semibold text-sm">{name}</h3>
      <span className="text-xs font-mono text-lightMuted">{source}</span>
    </div>
    <div className="text-primary font-mono text-2xl font-bold mb-1">{records}</div>
    <p className="text-lightMuted text-xs">{fields}</p>
  </motion.div>
);

export default function Research() {
  return (
    <>
      <Head>
        <title>Research — BRUT Framework</title>
      </Head>
      <TransitionEffect />

      {/* Header */}
      <Layout className="!pt-16 !pb-8">
        <div className="text-center max-w-3xl mx-auto">
          <AnimatedText
            text="Research & Validation"
            className="!text-6xl xl:!text-5xl md:!text-3xl"
          />
          <motion.p
            className="text-lightMuted text-lg mt-4 leading-relaxed md:text-base"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            Peer-reviewed publications, empirical validation against public databases,
            and 86 nights of continuous wearable sensor data.
          </motion.p>
        </div>
      </Layout>

      {/* Publications */}
      <Layout className="!py-8">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Publications</h2>
        </motion.div>

        <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
          <PaperCard
            number="I"
            title="S-Entropy and the Partition Capacity Axiom: A Mathematical Foundation for Categorical State Spaces"
            authors="K. Sachikonye"
            status="Preprint"
            delay={0}
            abstract="Introduces the foundational axiom C(n) = 2n² and derives the S-entropy coordinate system (S_k, S_t, S_e) for representing physiological states in a unified three-dimensional space. Establishes the Kuramoto order parameter framework and five partition regimes."
            tags={["partition-theory", "S-entropy", "Kuramoto", "foundations"]}
          />
          <PaperCard
            number="II"
            title="Cardiac Equations of State: Pressure-Volume Thermodynamics from Partition Boundary Conditions"
            authors="K. Sachikonye"
            status="Preprint"
            delay={1}
            abstract="Derives the Frank-Starling law, Windkessel model, and baroreflex as emergent properties of the partition equation of state PdV + VdP = C(n)kT. Classifies cardiac pathologies via Kuramoto regime boundaries, validated against MIT-BIH arrhythmia database."
            tags={["cardiac", "equations-of-state", "Frank-Starling", "hemodynamics"]}
          />
          <PaperCard
            number="III"
            title="Cardiac-Neural-Metabolic Integration: Cross-Scale Coherence and the Consciousness Window"
            authors="K. Sachikonye"
            status="Preprint"
            delay={2}
            abstract="Establishes the universal coupling law R_n/R_c = 0.87/√R_c, derives the consciousness window formula, and demonstrates REM active decoupling. Includes empirical validation across PhysioNet databases and Oura Ring sleep data."
            tags={["neural", "coupling", "consciousness", "sleep", "REM"]}
          />
          <PaperCard
            number="IV"
            title="Sensor Disambiguation via Partition-Coupled Metrics: From Single-Sensor Readings to Physiological State"
            authors="K. Sachikonye"
            status="In Preparation"
            delay={3}
            abstract="Introduces Partition-Coupled Heart Rate (PCHR), S-entropy health coordinates, Temperature-Corrected Coherence (TCC), and the Cross-Scale Coherence Index (CSCI) — novel composite metrics that leverage the partition framework to disambiguate identical sensor readings into distinct physiological states."
            tags={["wearables", "PCHR", "TCC", "CSCI", "disambiguation"]}
          />
        </div>
      </Layout>

      {/* Validation Datasets */}
      <div className="border-y border-primary/10">
        <Layout className="!py-12">
          <motion.div
            className="text-center mb-10"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="section-heading">Validation Datasets</h2>
            <p className="section-subheading mx-auto">
              Validated against multiple independent data sources spanning
              arrhythmia, heart failure, sleep, and longitudinal wearable recordings.
            </p>
          </motion.div>

          <div className="grid grid-cols-4 gap-6 xl:grid-cols-2 md:grid-cols-1">
            <DatasetCard
              name="MIT-BIH Arrhythmia"
              source="PhysioNet"
              records="48 records"
              fields="Beat-to-beat RR intervals, rhythm annotations, 1,439 analysis windows"
              delay={0}
            />
            <DatasetCard
              name="CHF RR Intervals"
              source="PhysioNet"
              records="15 records"
              fields="Long-term RR intervals from CHF patients, 1,399 analysis windows"
              delay={1}
            />
            <DatasetCard
              name="Sleep-EDF"
              source="PhysioNet"
              records="197 records"
              fields="Polysomnography: EEG, EOG, EMG, event markers, hypnograms"
              delay={2}
            />
            <DatasetCard
              name="Oura Ring"
              source="Personal"
              records="86 nights / 104 days"
              fields="5-min HR, RMSSD, hypnogram, temperature, SpO2, activity"
              delay={3}
            />
          </div>
        </Layout>
      </div>

      {/* Validation Panels */}
      <Layout className="!py-12">
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="section-heading">Validation Results</h2>
          <p className="section-subheading mx-auto">
            20 validation panels covering cardiac regime classification, neural
            coupling, metabolic integration, and sensor disambiguation.
          </p>
        </motion.div>

        <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
          <ValidationPanel
            title="Cardiac Regime Classification"
            delay={0}
            panels={[
              "Sleep stage R_c distributions (8,701 epochs)",
              "MIT-BIH rhythm regime mapping (1,439 windows)",
              "CHF vs NSR coherence deficit analysis",
              "Window-level cardiac landscape",
            ]}
          />
          <ValidationPanel
            title="Sleep Architecture"
            delay={1}
            panels={[
              "Transition matrix (86 nights)",
              "Sleep score correlations",
              "Stage-specific RMSSD profiles",
              "Activity-sleep coupling dynamics",
            ]}
          />
          <ValidationPanel
            title="Cardiac-Neural Coupling"
            delay={2}
            panels={[
              "R_c vs R_n per sleep stage",
              "Coupling formula validation (error = 0.011)",
              "REM active decoupling (gap = 0.375)",
              "EEG band profile analysis",
            ]}
          />
          <ValidationPanel
            title="Equations of State"
            delay={3}
            panels={[
              "Cardiac PV loop thermodynamics",
              "Frank-Starling derived curves",
              "Windkessel arterial compliance",
              "Regime sweep predictions",
            ]}
          />
          <ValidationPanel
            title="Sensor Disambiguation"
            delay={4}
            panels={[
              "PCHR decomposition across sleep stages",
              "S-entropy health coordinates mapping",
              "Temperature-corrected coherence (TCC)",
              "Cross-Scale Coherence Index (CSCI)",
            ]}
          />
          <ValidationPanel
            title="Predicted vs Measured"
            delay={5}
            panels={[
              "17 testable predictions status",
              "5 confirmed, 2 revised, 3 new discoveries",
              "Regime boundary calibration",
              "Cross-database consistency checks",
            ]}
          />
        </div>
      </Layout>

      {/* Key Numbers */}
      <div className="border-t border-primary/10">
        <Layout className="!py-12">
          <div className="grid grid-cols-5 gap-6 text-center xl:grid-cols-3 md:grid-cols-2 sm:grid-cols-1">
            {[
              { value: "78.8%", label: "AFIB epoch classification accuracy" },
              { value: "33.2", label: "Cohen's d: AFIB vs NSR" },
              { value: "0.011", label: "Coupling formula error (N1/N2)" },
              { value: "0.375", label: "REM cardiac-neural gap" },
              { value: "0.797", label: "CHF paradox R_c (> NSR 0.710)" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                className="glass-card"
                custom={i}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
              >
                <div className="text-2xl font-bold gradient-text font-mono">
                  {stat.value}
                </div>
                <div className="text-lightMuted text-xs mt-1">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </Layout>
      </div>
    </>
  );
}
